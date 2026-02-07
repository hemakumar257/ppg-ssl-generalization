from preprocessing.pipeline import PreprocessingPipeline
import traceback
import sys

try:
    pipeline = PreprocessingPipeline(output_dir="test_output")
    pipeline.run(datasets=['bidmc'])
except ZeroDivisionError:
    traceback.print_exc()
except Exception as e:
    print(f"Caught other exception: {e}")
    traceback.print_exc()
