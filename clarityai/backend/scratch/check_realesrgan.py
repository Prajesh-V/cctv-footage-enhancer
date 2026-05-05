import sys
try:
    from realesrgan import RealESRGANer
    import inspect
    print("RealESRGANer __init__ signature:")
    print(inspect.signature(RealESRGANer.__init__))
except ImportError as e:
    print(f"ImportError: {e}")
try:
    from gfpgan import GFPGANer
    print("\nGFPGANer __init__ signature:")
    print(inspect.signature(GFPGANer.__init__))
except ImportError as e:
    print(f"\nImportError (GFPGANer): {e}")
except Exception as e:
    print(f"\nError (GFPGANer): {e}")
