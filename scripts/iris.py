# Iris 3B support for Forge Neo
# Entry point: Forge Neo imports every scripts/*.py at startup, before the UI and before any model loads, which is
# when the model gets registered in the checkpoint list. Everything runs in the module scope; any failure leaves
# Forge Neo unchanged and says so in the console.

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENABLED = False

try:
    from forge_iris.native import compat

    _problems = compat.check()
    if _problems:
        compat.report(_problems)
    else:
        from forge_iris.native import patches

        patches.apply()
        ENABLED = True
except Exception:
    print("[Iris 3B] failed to enable, Forge Neo is unchanged:")
    traceback.print_exc()


if ENABLED:
    from modules import scripts

    from forge_iris import script as iris_script

    class IrisScript(scripts.Script):
        sorting_priority = 10
        create_group = False

        def title(self):
            return "Iris 3B"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            if not is_img2img:
                return []
            import gradio as gr

            with gr.Accordion("Iris 3B", open=False):
                mode = gr.Radio(list(iris_script.MODES), value=iris_script.MODE_IMAGE, label="Task",
                                elem_id="iris_task_mode")
                gr.Markdown("**Restore x4**: one pass with the upscaler LoRA in the prompt "
                            "(`<lora:iris-3b-upscaler_r128:1>`) or the upscaler checkpoint; the output is 4x the "
                            "input. **Depth**: one pass with the Iris depth checkpoint. Both ignore the prompt text, "
                            "steps and CFG.")
            return [mode]

        def before_process(self, p, *args):
            iris_script.script_before_process(p, *args[:1])

        def process_before_every_sampling(self, p, *args, **kwargs):
            iris_script.before_sampling(p, kwargs.get("noise"), kwargs.get("x"))

        def postprocess_image(self, p, pp, *args):
            iris_script.postprocess_image(p, pp)

        def postprocess(self, p, processed, *args):
            iris_script.postprocess(p)
