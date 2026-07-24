"""
batch_runner.py — Core batch video generation logic for faceless-gen.

Loops through a list of prompts, runs the full LangGraph pipeline for each,
exports final MP4 files to an output directory, and aggressively frees GPU/CPU
memory between iterations to prevent OOM errors on long batch runs (30+ videos).

Usage (programmatic):
    from backend.pipeline.batch_runner import run_batch

    results = run_batch(
        prompts=["Black holes explained", "How volcanoes form", ...],
        output_dir="outputs/batch",
    )
"""

import gc
import logging
import secrets
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VRAM / memory cleanup helpers
# ---------------------------------------------------------------------------

def _free_vram(unload_model: bool = False) -> None:
    """
    Aggressively free GPU and CPU memory between pipeline runs.

    Args:
        unload_model: If True, also destroys the class-level Wan I2V pipeline
                      cache (LocalWanBackend._pipeline). This fully releases
                      model weights from VRAM but forces a slow reload on the
                      next iteration. Use for 30+ video batches or if OOM
                      errors occur despite cache.cuda.empty_cache().
    """
    # Step 1: Python garbage collection
    gc.collect()

    # Step 2: CUDA cache flush (works on Kaggle/Colab GPU environments)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("_free_vram: torch.cuda.empty_cache() called")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon MPS — no explicit cache clear API, gc.collect() is best
            logger.info("_free_vram: MPS device — gc.collect() applied")
    except ImportError:
        logger.info("_free_vram: torch not available, skipping GPU cleanup")

    # Step 3 (optional): Destroy the Wan I2V pipeline to reclaim model VRAM
    if unload_model:
        try:
            from backend.providers.video_backend import LocalWanBackend
            if LocalWanBackend._pipeline is not None:
                del LocalWanBackend._pipeline
                LocalWanBackend._pipeline = None
                gc.collect()
                # Second CUDA flush after model deletion
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                logger.info("_free_vram: Wan I2V pipeline unloaded from memory")
        except Exception as exc:
            logger.warning("_free_vram: could not unload Wan pipeline: %s", exc)


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def run_batch(
    prompts: list[str],
    output_dir: str = "outputs/batch",
    stop_on_error: bool = False,
    unload_model_between_runs: bool = False,
) -> list[dict]:
    """
    Run the full LangGraph pipeline for each prompt and export final videos.

    Pipeline per prompt (Steps 1–5):
        scripting_node  → audio_node  → image_gen_node
        → video_node    → assembly_node  → final.mp4

    Args:
        prompts: List of topic/prompt strings to generate videos for.
        output_dir: Directory where exported MP4 files will be written.
                    Files are named video_01.mp4, video_02.mp4, etc.
        stop_on_error: If True, abort the entire batch on the first pipeline
                       error. Default is False (log error and continue).
        unload_model_between_runs: If True, fully unload the Wan I2V model from
                                   GPU memory between runs. Prevents OOM on
                                   very long batches but adds reload time per
                                   iteration (~2–5 min on first use, varies).

    Returns:
        List of result dicts, one per prompt, with keys:
            index       (int)          1-based index in the batch
            prompt      (str)          the original prompt string
            job_id      (str)          auto-generated hex job ID
            output_path (str | None)   absolute path to the exported MP4, or None
            error       (str | None)   error message if the run failed, else None
            duration_s  (float)        wall-clock seconds for this run
    """
    # Import here to avoid circular import at module level
    from backend.pipeline.graph import compiled_graph
    from backend.pipeline.state import initial_state

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total = len(prompts)
    results: list[dict] = []

    print(f"\n{'='*65}")
    print(f"  faceless-gen BATCH MODE  —  {total} videos to generate")
    print(f"  Output directory: {out_path.resolve()}")
    print(f"{'='*65}\n")

    batch_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        job_id = secrets.token_hex(6)
        run_start = time.perf_counter()

        print(f"{'─'*65}")
        print(f"  [{i+1:02d}/{total:02d}]  job_id={job_id}")
        print(f"  Prompt : {prompt}")
        print(f"{'─'*65}")

        result: dict = {
            "index": i + 1,
            "prompt": prompt,
            "job_id": job_id,
            "output_path": None,
            "error": None,
            "duration_s": 0.0,
        }

        try:
            state = initial_state(job_id=job_id, user_prompt=prompt)
            final_state = compiled_graph.invoke(state)

            # Print per-step progress log
            for entry in final_state.get("progress_log", []):
                print(f"    {entry}")

            pipeline_error = final_state.get("error")
            if pipeline_error:
                result["error"] = pipeline_error
                print(f"\n  ✗  Pipeline error: {pipeline_error}")
                if stop_on_error:
                    results.append(result)
                    print(f"\n  [BATCH] stop_on_error=True — aborting batch.")
                    break
            else:
                src = final_state.get("final_output")
                if src and Path(src).exists():
                    dest = out_path / f"video_{i+1:02d}.mp4"
                    shutil.copy2(src, dest)
                    result["output_path"] = str(dest.resolve())
                    print(f"\n  ✓  Exported → {dest.resolve()}")
                else:
                    result["error"] = (
                        "Pipeline completed but final_output path is missing or not found. "
                        f"(final_output={src!r})"
                    )
                    print(f"\n  ⚠  {result['error']}")

        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"\n  ✗  Unexpected exception: {result['error']}")
            logger.exception("batch_runner: unhandled exception on prompt %d", i + 1)
            if stop_on_error:
                results.append(result)
                print(f"\n  [BATCH] stop_on_error=True — aborting batch.")
                break

        finally:
            elapsed = time.perf_counter() - run_start
            result["duration_s"] = round(elapsed, 1)
            results.append(result)

            # --- VRAM / memory cleanup ---
            print(
                f"\n  [VRAM] Cleaning up memory after job {job_id} "
                f"(unload_model={unload_model_between_runs})..."
            )
            _free_vram(unload_model=unload_model_between_runs)
            print(f"  [VRAM] Done.  Run time: {elapsed:.1f}s\n")

    # -----------------------------------------------------------------------
    # Batch summary
    # -----------------------------------------------------------------------
    total_elapsed = time.perf_counter() - batch_start
    success_runs = [r for r in results if r["output_path"] is not None]
    failed_runs  = [r for r in results if r["error"] is not None]

    print(f"\n{'='*65}")
    print("  BATCH COMPLETE")
    print(f"{'='*65}")
    print(f"  Processed : {len(results)}/{total}")
    print(f"  Succeeded : {len(success_runs)}")
    print(f"  Failed    : {len(failed_runs)}")
    print(f"  Wall time : {total_elapsed/60:.1f} min")
    print()
    for r in results:
        status = "✓" if r["output_path"] else "✗"
        prompt_preview = r["prompt"][:55] + "…" if len(r["prompt"]) > 55 else r["prompt"]
        print(f"  [{status}] {r['index']:02d}. {prompt_preview}  ({r['duration_s']:.0f}s)")
        if r["error"]:
            print(f"       ↳ Error: {r['error'][:90]}")
    print(f"{'='*65}\n")

    return results
