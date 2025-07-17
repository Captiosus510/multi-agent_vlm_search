#!/usr/bin/env python3
"""
Comprehensive comparison of different vision-language models for semantic mapping.
Includes performance benchmarks and recommendations for real-time applications.
"""

import time
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import torch

# Import your existing models
from llm_search.utils.blip2 import BLIP2ITM
from llm_search.utils.siglip import SigLipInterface


class VLMBenchmark:
    """Benchmark different Vision-Language Models for semantic mapping."""
    
    def __init__(self):
        """Initialize all available models."""
        self.models = {}
        
        # Initialize models with error handling
        try:
            self.models['blip2'] = BLIP2ITM()
            print("✓ BLIP2 loaded successfully")
        except Exception as e:
            print(f"✗ BLIP2 failed to load: {e}")
            
        try:
            self.models['siglip'] = SigLipInterface()
            print("✓ SigLIP loaded successfully")
        except Exception as e:
            print(f"✗ SigLIP failed to load: {e}")
            
        # try:
        #     self.models['miniclip'] = MiniCLIPInterface()
        #     print("✓ MiniCLIP loaded successfully")
        # except Exception as e:
        #     print(f"✗ MiniCLIP failed to load: {e}")
    
    def benchmark_inference_speed(self, image_path: str, text: str, num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark inference speed for all models.
        
        Args:
            image_path: Path to test image
            text: Text prompt for comparison
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with model names and average inference times
        """
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        
        results = {}
        
        for model_name, model in self.models.items():
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                
                if model_name == 'blip2':
                    _ = model.cosine(image_array, text)
                elif model_name == 'siglip':
                    _ = model.compute_confidence(image_array, [text])
                elif model_name == 'miniclip':
                    _ = model.cosine(image_array, text)
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[model_name] = np.mean(times)
            print(f"{model_name}: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
        
        return results
    
    def benchmark_accuracy(self, test_cases: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """
        Benchmark accuracy across different test cases.
        
        Args:
            test_cases: List of (image_path, text, expected_score) tuples
            
        Returns:
            Dictionary with model names and accuracy scores
        """
        results = {}
        
        for model_name, model in self.models.items():
            errors = []
            
            for image_path, text, expected in test_cases:
                image = Image.open(image_path).convert("RGB")
                image_array = np.array(image)
                predicted = 0.0
                if model_name == 'blip2':
                    predicted = model.cosine(image_array, text)
                elif model_name == 'siglip':
                    predicted = model.compute_confidence(image_array, [text])
                elif model_name == 'miniclip':
                    predicted = model.cosine(image_array, text)
                
                error = abs(predicted - expected)
                errors.append(error)
            
            results[model_name] = np.mean(errors)
            print(f"{model_name} average error: {np.mean(errors):.4f}")
        
        return results


def get_recommendations() -> str:
    """
    Get recommendations for different use cases.
    
    Returns:
        Formatted recommendations string
    """
    recommendations = """
    🚀 RECOMMENDATIONS FOR REAL-TIME SEMANTIC MAPPING:
    
    1. **For Maximum Speed (Real-time robotics):**
       - Primary: SigLIP (your current choice) ✓
       - Backup: MiniCLIP with patch32
       - Target: >30 FPS on modern GPU
    
    2. **For Balanced Speed/Accuracy:**
       - Primary: SigLIP with temperature scaling
       - Secondary: MiniCLIP with patch16
       - Target: 10-30 FPS
    
    3. **For Maximum Accuracy (Offline processing):**
       - Primary: BLIP2 (your current implementation)
       - Secondary: Larger SigLIP models
       - Target: <10 FPS acceptable
    
    4. **Memory Optimization:**
       - Use model quantization (int8/fp16)
       - Batch processing for multiple queries
       - Model pruning for edge deployment
    
    5. **Real-time Optimization Tips:**
       - Precompute text embeddings for fixed queries
       - Use image resizing/cropping for speed
       - Implement model caching
       - Consider TensorRT optimization
    
    📊 PERFORMANCE COMPARISON (Approximate):
    
    Model           | Speed (FPS) | Accuracy | Memory (GB) | Best Use Case
    ---------------|-------------|----------|-------------|---------------
    SigLIP         | 50-100      | High     | 2-4         | Real-time mapping ✓
    MiniCLIP       | 30-80       | Medium   | 1-2         | Edge devices
    BLIP2          | 5-15        | Highest  | 4-8         | Offline analysis
    
    🎯 FOR YOUR SEMANTIC MAPPING:
    
    Your current SigLIP implementation is excellent for real-time semantic mapping!
    Consider these enhancements:
    
    1. Multi-scale processing (different image resolutions)
    2. Temporal consistency (smooth confidence over time)
    3. Ensemble methods (combine multiple models)
    4. Adaptive thresholding based on scene complexity
    """
    
    return recommendations


def main():
    """Run benchmarks and show recommendations."""
    print("🔍 Vision-Language Model Benchmark for Semantic Mapping")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = VLMBenchmark()
    
    # Test with your existing image
    test_image = "llm_search/utils/test_images/silver_cat.jpg"
    test_text = "a photo of a cat"
    
    print(f"\n⏱️  Speed Benchmark (Image: {test_image}, Text: '{test_text}'):")
    print("-" * 40)
    
    try:
        speed_results = benchmark.benchmark_inference_speed(test_image, test_text)
        
        # Sort by speed (fastest first)
        sorted_results = sorted(speed_results.items(), key=lambda x: x[1])
        
        print("\n🏃 Speed Ranking (Fastest to Slowest):")
        for i, (model, time_taken) in enumerate(sorted_results, 1):
            fps = 1.0 / time_taken
            print(f"{i}. {model.upper()}: {fps:.1f} FPS ({time_taken:.4f}s)")

        # Show accuracy results
        print("\n📊 Accuracy Results:")
        for model, accuracy in benchmark.benchmark_accuracy([(test_image, test_text, 1)]).items():
            print(f"{model.upper()}: {accuracy:.2f}")

    except Exception as e:
        print(f"Speed benchmark failed: {e}")
    
    # Show recommendations
    # print("\n" + get_recommendations())
    
    # Memory usage tips
    print("\n💡 MEMORY OPTIMIZATION TIPS:")
    print("- Use torch.cuda.empty_cache() periodically")
    print("- Enable mixed precision training (fp16)")
    print("- Consider model quantization for deployment")
    print("- Use gradient checkpointing for large models")
    
    # Real-time implementation suggestions
    print("\n🔧 REAL-TIME IMPLEMENTATION SUGGESTIONS:")
    print("1. Precompute text embeddings for fixed goals")
    print("2. Use image preprocessing queues")
    print("3. Implement confidence smoothing over time")
    print("4. Add early stopping for obvious cases")
    print("5. Use multiple models in ensemble for critical decisions")


if __name__ == "__main__":
    main()
