"""
Checkpoint loading for the WGAN-GP checkpoints, which pickle references to
modules the state dict itself does not need. Two independent mechanisms live
here.

1. `_missing_module_stubs()` -- the path that runs on EVERY `load_checkpoint`.
   A checkpoint pickles its Lightning `hyper_parameters`, one of which
   (`penalty_model`) is a class reference into `physics.penalty_models`.
   Unpickling must resolve that name even though only
   `checkpoint['state_dict']` is ever read, so the class object is discarded
   the moment it is built. The context manager registers an exact-named
   permissive stub in `sys.modules`, only when `importlib.util.find_spec`
   reports the module genuinely ABSENT (a module that exists but fails to import raises
   instead of being silently masked), and pops it again in `finally` so
   nothing leaks into the interpreter's module table. In practice
   `physics/penalty_models.py` exists, so `find_spec` finds the real module and
   the stub never fires; it stays in place in case a checkpoint ever
   references a module that genuinely does not exist.

2. `rescue_checkpoint()` / `MockFinder` -- a heavier fallback for checkpoints
   referencing a capital-`Physics` package under a different name. It installs
   a prefix-matching `sys.meta_path` finder, extracts the state_dict, and
   writes a clean `.pth` sidecar loadable without the hook. `load_checkpoint`
   only reaches it if the direct load fails.

The two are deliberately separate: `MockFinder` prefix-matches and sits at the
head of `sys.meta_path`, so widening it to lowercase "physics" would shadow
this package's real `physics.*` modules.
"""
import sys
import torch
import types
import logging
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec


# --- 1. Define the Universal Dummy Objects ---
class UniversalDummyObject:
    """A generic object that swallows all attribute access and calls."""

    def __init__(self, *args, **kwargs): pass

    def __call__(self, *args, **kwargs): return self

    def __getattr__(self, key): return self

    def __setstate__(self, state): pass


class MockModule(types.ModuleType):
    """A fake module that returns a dummy object for ANY attribute access."""

    def __getattr__(self, key):
        return UniversalDummyObject()

    __all__ = []  # Satisfy 'from module import *'


# --- 2. Define the Import Hook (The Fix) ---
class MockLoader(Loader):
    def create_module(self, spec):
        # Create a MockModule instead of a standard module
        module = MockModule(spec.name)
        # Mark it as a package so sub-imports (like Physics.penalty_models) work
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


class MockFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # Intercept ANY import starting with 'Physics'
        if fullname.startswith("Physics"):
            logging.info(f" -> Mocking import: {fullname}")
            return ModuleSpec(fullname, MockLoader())
        return None


# --- 3. The Recovery Function ---
def rescue_checkpoint(ckpt_path: Path, target_path: Path):
    logging.info(f"Attempting to rescue: {ckpt_path}")

    # Install the hook into Python's import system
    if not any(isinstance(x, MockFinder) for x in sys.meta_path):
        sys.meta_path.insert(0, MockFinder())
        logging.info(" -> Import hook installed.")

    try:
        # Load the checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        logging.info("\nSUCCESS: Checkpoint loaded into memory!")

        # Extract the state_dict
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint  # It might be the dict itself
        elif hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()

        if state_dict:
            torch.save(state_dict, target_path)
            logging.info(f" -> Clean weights saved to: {target_path}")
            logging.info(" -> You can now load this file normally without the 'Physics' module.")
        else:
            logging.info(" -> Error: Could not locate state_dict inside the loaded object.")

    except Exception as e:
        logging.info(f"\nFailed with error: {e}")
        import traceback
        traceback.print_exc()


# --- 3b. Stub modules for hyper-parameter class references a checkpoint
# might pickle but that the running interpreter does not have ---
# A checkpoint pickles its Lightning `hyper_parameters`, one of which
# (`penalty_model`) is a CLASS REFERENCE into `physics.penalty_models`. We
# only ever read `checkpoint['state_dict']`, so the class itself is discarded
# immediately; it just has to resolve during unpickling. `MockFinder` above
# cannot be reused: it prefix-matches and inserts itself at the head of
# `sys.meta_path`, so widening it to "physics" would shadow this package's
# real `physics.*` modules. Instead register exact module names in
# `sys.modules`, and only when the real module is genuinely ABSENT, so a real
# module always wins -- and one that exists but fails to import raises its own error rather
# than being masked.
_MISSING_MODULES = ("physics.penalty_models",)


@contextmanager
def _missing_module_stubs():
    """Temporarily resolve module paths a checkpoint's hyper-parameters
    reference but that do not exist to permissive stubs."""
    class _Stub:
        def __init__(self, *args, **kwargs): pass

        def __setstate__(self, state): pass

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            return type(name, (_Stub,), {})

    installed = []
    for name in _MISSING_MODULES:
        if name in sys.modules:
            continue
        # `find_spec` asks "does this module EXIST?" without executing it, so a
        # module that exists but raises on import propagates its own error at
        # unpickle time instead of being silently replaced by a stub -- which a
        # bare `except ImportError` around `import_module` would have done.
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None  # a parent package is missing => the module is absent
        if spec is not None:
            continue     # the real module exists; let the import system use it
        sys.modules[name] = _StubModule(name)
        installed.append(name)
        logging.info(f" -> Stubbed missing module: {name}")
    try:
        yield
    finally:
        for name in installed:
            sys.modules.pop(name, None)


def load_checkpoint(ckpt_path: Path) -> dict:
    # Start with a naive loading
    try:
        with _missing_module_stubs():
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        logging.info("\nCheckpoint loaded without mocking (unexpected success).")
        return checkpoint['state_dict']
    except Exception as e:
        logging.info(f"\nInitial load failed with error: {e}")

        if ckpt_path.with_suffix(".pth").exists():
            clean_checkpoint = torch.load(ckpt_path.with_suffix(".pth"), map_location='cpu', weights_only=False)
            logging.info("\nClean checkpoint already exists and loaded successfully!")
            return clean_checkpoint

        logging.info(" -> Proceeding to rescue with import hook...")
        rescue_checkpoint(ckpt_path, ckpt_path.with_suffix(".pth"))

        # After rescue, try loading the cleaned checkpoint
        try:
            clean_checkpoint = torch.load(ckpt_path.with_suffix(".pth"), map_location='cpu')
            logging.info("\nClean checkpoint loaded successfully after rescue!")
            return clean_checkpoint
        except Exception as e:
            logging.info(f"\nFailed to load clean checkpoint: {e}")
            raise RuntimeError("Rescue failed. Check the logs for details.") from e