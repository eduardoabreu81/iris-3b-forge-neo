# Iris 3B support for Forge Neo
# The generation error mechanism: Forge swallows exceptions raised in script callbacks, so an error set here in
# before_process is re-raised from inside the model (which Forge calls directly) and the generation dies with the
# right message instead of producing a wrong image.

_pending_error = None


class IrisError(RuntimeError):
    """A message the user can act on."""


def set_pending_error(error):
    global _pending_error
    _pending_error = error


def raise_pending_error():
    if _pending_error is not None:
        raise _pending_error


# The single-pass task of the current generation (tasks.RESTORE, tasks.DEPTH or None), set by the script before
# Forge loads the model and read by the DiT's forward.
_task = None


def set_task(task):
    global _task
    _task = task


def current_task():
    return _task
