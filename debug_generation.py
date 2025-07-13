"""
Complete debugging tool for VLM Bridge generation issues.

This script provides comprehensive diagnostics for the caption generation process,
helping identify why training metrics improve but generation quality fails.
"""

import torch
import os
from pathlib import Path
import json
from typing import Dict, Any, Optional

# Set environment to reduce noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

from vlm_bridge.model_architecture import FullModel
from vlm_bridge.data_pipeline.data_loader import VLDataset


class GenerationDebugger:
    """Comprehensive generation debugging and analysis tool."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize debugger with optional trained model."""
        print("🔧 Initializing Generation Debugger...")

        # Load model
        self.model = FullModel()

        if model_path and Path(model_path).exists():
            print(f"📁 Loading trained model from: {model_path}")
            self.model.load_model(model_path)
        else:
            print("⚠️  Using untrained model for baseline testing")

        self.model.eval()

        # Store debugging info
        self.debug_info = {
            "generation_steps": [],
            "attention_weights": [],
            "logits_stats": [],
            "token_analysis": [],
        }

        print("✅ Debugger initialized!")

    def debug_single_generation(
        self, image: torch.Tensor, max_length: int = 20, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Debug a single generation with detailed step-by-step analysis.

        Returns comprehensive debugging information.
        """
        print(f"\n🔍 Starting detailed generation debug (max_length={max_length})")

        debug_session = {
            "steps": [],
            "final_result": None,
            "issues_detected": [],
            "recommendations": [],
        }

        with torch.no_grad():
            # Prepare image
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.model.device)

            # Extract vision features once
            vision_features = self.model.vision_encoder(image)
            if verbose:
                print(f"📸 Vision features: {vision_features.shape}")

            # Initialize with BOS token
            tokenizer = self.model.language_model.tokenizer
            input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(self.model.device)

            if verbose:
                print(
                    f"🚀 Starting with BOS token: {tokenizer.bos_token_id} ('{tokenizer.bos_token}')"
                )

            # Generation loop with detailed debugging
            for step in range(max_length):
                step_info = {
                    "step": step,
                    "input_sequence": input_ids.clone(),
                    "input_tokens": tokenizer.decode(input_ids[0]).strip(),
                    "sequence_length": input_ids.shape[1],
                }

                # Get current text embeddings
                text_embeddings = self.model.language_model.get_embeddings(input_ids)
                step_info["text_embeddings_shape"] = text_embeddings.shape
                step_info["text_embeddings_stats"] = {
                    "mean": text_embeddings.mean().item(),
                    "std": text_embeddings.std().item(),
                    "min": text_embeddings.min().item(),
                    "max": text_embeddings.max().item(),
                }

                # Create attention mask
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

                # Enhance with vision through Bridge
                enhanced_embeddings = self.model.bridge_module(
                    vision_features, text_embeddings
                )
                step_info["enhanced_embeddings_shape"] = enhanced_embeddings.shape
                step_info["enhanced_embeddings_stats"] = {
                    "mean": enhanced_embeddings.mean().item(),
                    "std": enhanced_embeddings.std().item(),
                    "min": enhanced_embeddings.min().item(),
                    "max": enhanced_embeddings.max().item(),
                }

                # Check for numerical issues
                if torch.isnan(enhanced_embeddings).any():
                    debug_session["issues_detected"].append(
                        f"Step {step}: NaN in enhanced_embeddings"
                    )
                if torch.isinf(enhanced_embeddings).any():
                    debug_session["issues_detected"].append(
                        f"Step {step}: Inf in enhanced_embeddings"
                    )

                # Generate logits
                logits = self.model.language_model.forward_from_embeddings(
                    enhanced_embeddings, attention_mask=attention_mask
                )

                # Analyze logits
                next_token_logits = logits[:, -1, :]
                step_info["logits_shape"] = next_token_logits.shape
                step_info["logits_stats"] = {
                    "mean": next_token_logits.mean().item(),
                    "std": next_token_logits.std().item(),
                    "min": next_token_logits.min().item(),
                    "max": next_token_logits.max().item(),
                }

                # Get top-k predictions for analysis
                top_k = torch.topk(next_token_logits, k=5, dim=-1)
                step_info["top_5_logits"] = top_k.values[0].tolist()
                step_info["top_5_tokens"] = top_k.indices[0].tolist()
                step_info["top_5_words"] = [
                    tokenizer.decode([token_id])
                    for token_id in top_k.indices[0].tolist()
                ]

                # Check for extreme logits
                if next_token_logits.max() > 100:
                    debug_session["issues_detected"].append(
                        f"Step {step}: Extremely high logits (max: {next_token_logits.max():.2f})"
                    )
                if next_token_logits.min() < -100:
                    debug_session["issues_detected"].append(
                        f"Step {step}: Extremely low logits (min: {next_token_logits.min():.2f})"
                    )

                # Greedy selection for debugging
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                next_token_id = next_token.item()
                next_word = tokenizer.decode([next_token_id])

                step_info["selected_token_id"] = next_token_id
                step_info["selected_word"] = next_word

                if verbose:
                    print(
                        f"  Step {step:2d}: '{next_word}' (ID: {next_token_id:5d}) | "
                        f"Logit: {next_token_logits[0, next_token_id].item():6.2f} | "
                        f"Top-5: {[f'{w}({v:.1f})' for w, v in zip(step_info['top_5_words'][:3], step_info['top_5_logits'][:3])]}"
                    )

                # Check for immediate EOS
                if next_token_id == tokenizer.eos_token_id:
                    step_info["eos_triggered"] = True
                    if verbose:
                        print(f"🛑 EOS token triggered at step {step}")
                    if step == 0:
                        debug_session["issues_detected"].append(
                            "EOS triggered immediately after BOS"
                        )
                    break
                else:
                    step_info["eos_triggered"] = False

                # Append token to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Store step info
                debug_session["steps"].append(step_info)

                # Check for repetition
                if len(debug_session["steps"]) >= 3:
                    last_3_tokens = [
                        s["selected_token_id"] for s in debug_session["steps"][-3:]
                    ]
                    if len(set(last_3_tokens)) == 1:
                        debug_session["issues_detected"].append(
                            f"Token repetition detected: {last_3_tokens}"
                        )

            # Decode final result
            final_caption = tokenizer.decode(
                input_ids[0], skip_special_tokens=True
            ).strip()
            debug_session["final_result"] = {
                "caption": final_caption,
                "token_count": len(debug_session["steps"]),
                "final_sequence": input_ids[0].tolist(),
                "raw_decoded": tokenizer.decode(input_ids[0]).strip(),
            }

            if verbose:
                print(f"\n📝 Final caption: '{final_caption}'")
                print(f"📊 Token count: {len(debug_session['steps'])}")

        # Generate recommendations
        if not debug_session["steps"]:
            debug_session["recommendations"].append(
                "No tokens generated - check EOS immediate trigger"
            )
        elif len(set([s["selected_token_id"] for s in debug_session["steps"]])) <= 2:
            debug_session["recommendations"].append(
                "Low token diversity - check sampling strategy"
            )

        return debug_session

    def test_multiple_strategies(self, image: torch.Tensor) -> Dict[str, Any]:
        """Test multiple generation strategies and compare results."""
        print("\n🧪 Testing multiple generation strategies...")

        strategies = {
            "greedy": {"do_sample": False},
            "low_temp": {"temperature": 0.1, "do_sample": True, "top_p": 1.0},
            "medium_temp": {"temperature": 0.7, "do_sample": True, "top_p": 0.9},
            "high_temp": {"temperature": 1.0, "do_sample": True, "top_p": 0.9},
            "no_top_p": {"temperature": 0.7, "do_sample": True, "top_p": 1.0},
        }

        results = {}

        for strategy_name, params in strategies.items():
            print(f"  Testing {strategy_name}: {params}")
            try:
                caption = self.model.generate_caption(image, max_length=30, **params)
                results[strategy_name] = {
                    "caption": caption,
                    "length": len(caption.split()) if caption else 0,
                    "params": params,
                    "success": len(caption) > 0,
                }
                print(f"    Result: '{caption}' ({len(caption.split())} words)")
            except Exception as e:
                results[strategy_name] = {
                    "caption": "",
                    "length": 0,
                    "params": params,
                    "success": False,
                    "error": str(e),
                }
                print(f"    Error: {e}")

        return results

    def compare_with_without_bridge(self, image: torch.Tensor) -> Dict[str, Any]:
        """Compare generation with and without Bridge module."""
        print("\n🔀 Comparing generation with/without Bridge...")

        results = {}

        # Test with Bridge (normal)
        print("  Testing WITH Bridge...")
        try:
            with_bridge = self.model.generate_caption(
                image, max_length=20, do_sample=False
            )
            results["with_bridge"] = {
                "caption": with_bridge,
                "length": len(with_bridge.split()) if with_bridge else 0,
            }
            print(f"    Result: '{with_bridge}'")
        except Exception as e:
            results["with_bridge"] = {"caption": "", "length": 0, "error": str(e)}
            print(f"    Error: {e}")

        # Test without Bridge (direct text embeddings)
        print("  Testing WITHOUT Bridge...")
        try:
            # Temporarily modify the model to skip Bridge
            original_forward = self.model.bridge_module.forward

            def bypass_bridge(vision_features, text_embeddings):
                # Return text embeddings unchanged
                return text_embeddings

            self.model.bridge_module.forward = bypass_bridge

            without_bridge = self.model.generate_caption(
                image, max_length=20, do_sample=False
            )
            results["without_bridge"] = {
                "caption": without_bridge,
                "length": len(without_bridge.split()) if without_bridge else 0,
            }
            print(f"    Result: '{without_bridge}'")

            # Restore original forward
            self.model.bridge_module.forward = original_forward

        except Exception as e:
            results["without_bridge"] = {"caption": "", "length": 0, "error": str(e)}
            print(f"    Error: {e}")
            # Ensure we restore the original forward
            self.model.bridge_module.forward = original_forward

        return results


def main():
    """Run comprehensive generation debugging."""
    print("🚨 VLM Bridge Generation Debugging Session")
    print("=" * 60)

    # Initialize debugger
    debugger = GenerationDebugger()

    # Load test image
    data_dir = Path("data/groundcap/train")
    if not data_dir.exists():
        print("❌ No test data found. Please run data pipeline first.")
        return

    # Get a sample image
    dataset = VLDataset(str(data_dir))
    sample = dataset[0]  # First sample
    image = sample["images"]

    print(f"🖼️  Testing with sample image: {sample.get('image_path', 'unknown')}")

    # Run comprehensive debugging
    print("\n" + "=" * 60)
    print("Phase 1: Detailed Step-by-Step Analysis")
    debug_result = debugger.debug_single_generation(image, max_length=15, verbose=True)

    print("\n" + "=" * 60)
    print("Phase 2: Multiple Strategy Testing")
    strategy_results = debugger.test_multiple_strategies(image)

    print("\n" + "=" * 60)
    print("Phase 3: Bridge vs No-Bridge Comparison")
    bridge_comparison = debugger.compare_with_without_bridge(image)

    # Summary
    print("\n" + "=" * 60)
    print("🎯 DEBUGGING SUMMARY")
    print("=" * 60)

    if debug_result["issues_detected"]:
        print("❌ Issues detected:")
        for issue in debug_result["issues_detected"]:
            print(f"  - {issue}")
    else:
        print("✅ No obvious issues detected in generation process")

    print("\n📊 Strategy comparison:")
    for strategy, result in strategy_results.items():
        status = "✅" if result["success"] else "❌"
        print(
            f"  {status} {strategy:12s}: {result['length']:2d} words - '{result['caption'][:50]}{'...' if len(result['caption']) > 50 else ''}'"
        )

    print("\n🔀 Bridge comparison:")
    print(
        f"  With Bridge:    {bridge_comparison['with_bridge']['length']:2d} words - '{bridge_comparison['with_bridge']['caption']}'"
    )
    print(
        f"  Without Bridge: {bridge_comparison['without_bridge']['length']:2d} words - '{bridge_comparison['without_bridge']['caption']}'"
    )

    # Save detailed results
    results_file = "debug_generation_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "debug_session": debug_result,
                "strategy_results": strategy_results,
                "bridge_comparison": bridge_comparison,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Detailed results saved to: {results_file}")
    print("\n🎯 Next steps: Analyze the results and implement targeted fixes!")


if __name__ == "__main__":
    main()
