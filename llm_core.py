import pandas as pd
import numpy as np
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import streamlit as st
import utils # Assuming utils.py is in the same directory

# Conditional import for vLLM
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    st.warning("vLLM or transformers not found. LLM features will be disabled. Please install required packages.")
    # Define dummy classes/functions if vLLM is not available to avoid runtime errors
    class LLM: pass
    class SamplingParams: pass
    class AutoTokenizer: pass


class LLMHandler:
    def __init__(self, config: Dict[str, Any]):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM and/or transformers are required but not installed.")

        self.config = config
        self.llm_settings = config.get('llm_settings', {})
        self.models_to_use = self.llm_settings.get('models_to_use', [])
        self.loading_strategy = self.llm_settings.get('loading_strategy', 'sequential')
        self.num_gpus = self.llm_settings.get('num_gpus', 1)
        self.gpu_mem_util = self.llm_settings.get('gpu_memory_utilization_per_model', 0.8)
        self.sampling_config = self.llm_settings.get('sampling_params', {})

        self.loaded_models = {} # Stores loaded LLM instances {model_id: llm_instance}
        self.tokenizers = {} # Stores loaded tokenizer instances {model_id: tokenizer_instance}

    def _load_model(self, model_id: str):
        # Loads a single LLM model and tokenizer if not already loaded.
        if model_id in self.loaded_models:
            return self.loaded_models[model_id], self.tokenizers[model_id]

        # Use st.write within the status context if available, otherwise print
        status_update_func = st.write # Default to st.write
        # Check if called within a status context (heuristic)
        # Note: Streamlit doesn't provide a direct way to check this easily.
        # We rely on app.py using st.status.
        status_update_func(f"⏳ Loading model: {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Adjust GPU memory utilization based on strategy and number of models
            gpu_util = self.gpu_mem_util
            if self.loading_strategy == 'concurrent' and len(self.models_to_use) > 0:
                 # Basic division - might need refinement based on actual model sizes
                 # Ensure it doesn't exceed a reasonable limit per GPU
                 gpu_util = min(self.gpu_mem_util, 0.95 / max(1, len(self.models_to_use)))


            llm = LLM(
                model=model_id,
                tokenizer=model_id, # Pass tokenizer explicitly if needed
                tensor_parallel_size=self.num_gpus,
                gpu_memory_utilization=gpu_util,
                # Add other vLLM options if needed, e.g., dtype='auto'
                # enforce_eager=True # Might help with some compatibility issues
                trust_remote_code=True # May be needed for some models like Phi
            )
            self.loaded_models[model_id] = llm
            self.tokenizers[model_id] = tokenizer
            status_update_func(f"✅ Model loaded: {model_id}")
            return llm, tokenizer
        except Exception as e:
            st.error(f"Failed to load model {model_id}: {e}")
            # Attempt to free memory if loading failed partially
            if model_id in self.loaded_models: del self.loaded_models[model_id]
            if model_id in self.tokenizers: del self.tokenizers[model_id]
            try:
                 import torch # Requires torch
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache() # Try to clear cache
            except ImportError:
                 pass # Ignore if torch is not available
            return None, None

    def _unload_model(self, model_id: str):
        # Unloads a model and clears GPU cache.
        if model_id in self.loaded_models:
            status_update_func = st.write # Default to st.write
            status_update_func(f"Unloading model: {model_id}...")
            try:
                # Explicitly delete the model and tokenizer objects
                llm_instance = self.loaded_models.pop(model_id, None)
                tokenizer_instance = self.tokenizers.pop(model_id, None)
                del llm_instance
                del tokenizer_instance

                # vLLM might not have an explicit unload, rely on garbage collection and cache clearing
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass # Ignore if torch is not available
                status_update_func(f"Model unloaded: {model_id}")
            except Exception as e:
                st.warning(f"Error during unloading {model_id}: {e}")


    def _generate_responses(self, model_id: str, prompt: str) -> List[str]:
        # Generates multiple responses (n) from a single model.
        responses = []
        llm, tokenizer = self._load_model(model_id)
        if not llm or not tokenizer:
            return ["Error: Model not loaded"] * self.sampling_config.get('n', 1) # Return error strings

        # Define stop tokens - rely on model's natural stop tokens or max_tokens.
        # Add common EOS tokens. The prompt strongly requests ending with the box.
        stop_tokens = ["<|eot_id|>", "<|endoftext|>", "### End Response", "\n\n\n\n"] # Common EOS tokens

        sampling_params = SamplingParams(
            n=self.sampling_config.get('n', 5),
            temperature=self.sampling_config.get('temperature', 0.6),
            top_p=self.sampling_config.get('top_p', 0.9),
            max_tokens=self.sampling_config.get('max_tokens', 1500),
            stop=stop_tokens # Use refined stop tokens
        )

        try:
            # Format prompt using chat template if available
            try:
                 messages = [{"role": "user", "content": prompt}]
                 # add_generation_prompt=True is important for instruction-tuned models
                 formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                 st.warning(f"Could not apply chat template for {model_id}. Using raw prompt.")
                 formatted_prompt = prompt # Fallback to raw prompt

            outputs = llm.generate(formatted_prompt, sampling_params)

            # Check if outputs is structured as expected
            if outputs and isinstance(outputs, list) and len(outputs) > 0:
                 request_output = outputs[0]
                 if hasattr(request_output, 'outputs') and isinstance(request_output.outputs, list):
                     for output in request_output.outputs:
                         if hasattr(output, 'text'):
                              # Strip leading/trailing whitespace from the raw response
                              responses.append(output.text.strip())
                         else:
                              responses.append("Error: Invalid output format (missing text)")
                 else:
                      responses = ["Error: Invalid output format (missing outputs list)"] * self.sampling_config.get('n', 1)
            else:
                 responses = ["Error: No output generated"] * self.sampling_config.get('n', 1)


        except Exception as e:
            st.error(f"Error during generation with {model_id}: {e}")
            import traceback
            st.code(traceback.format_exc()) # Show traceback for debugging
            responses = [f"Error during generation: {e}"] * self.sampling_config.get('n', 1) # Return error strings

        return responses

    # Modified to accept and update a progress bar
    def get_ensemble_recommendation(self, prompt: str, available_viz_options: List[str], progress_bar: Optional[st.delta_generator.DeltaGenerator] = None) -> Tuple[str, str, Dict[str, int], List[str]]:
        # Orchestrates getting recommendations from multiple models and performing majority voting.
        all_recommendations = []
        all_stories = []
        model_responses = {} # Store raw responses per model if needed for debugging

        if not self.models_to_use:
             st.error("No models specified in the configuration.")
             return "Error: No models configured", "Table", {}, []

        models_to_process = self.models_to_use
        loaded_concurrently = [] # Track models loaded in this run if concurrent

        # Load models concurrently if specified
        if self.loading_strategy == 'concurrent':
            st.write("Loading all specified models concurrently...")
            load_success = True
            for model_id in models_to_process:
                if model_id not in self.loaded_models:
                    llm, _ = self._load_model(model_id)
                    if not llm:
                         load_success = False
                         st.error(f"Failed to load {model_id} in concurrent mode. Check VRAM.")
                    else:
                         loaded_concurrently.append(model_id)
            if not load_success:
                 st.warning("Some models failed to load. Proceeding with loaded models, but results may be affected.")


        # Generate responses from each model
        total_models = len(models_to_process)
        # Use st.status for better progress indication (passed from app.py)
        # Progress bar is also passed from app.py

        for i, model_id in enumerate(models_to_process):
            # Update status within the loop (assuming called within st.status context)
            st.write(f"--- Starting: {model_id} (Model {i+1}/{total_models}) ---")

            # Load model if sequential strategy
            if self.loading_strategy == 'sequential':
                 llm, _ = self._load_model(model_id)
                 if not llm:
                     st.warning(f"Skipping model {model_id} due to loading error.")
                     st.write(f"--- Skipped: {model_id} ---")
                     # Update progress bar even if skipped
                     if progress_bar:
                          progress_bar.progress((i + 1) / total_models)
                     continue # Skip to next model if loading failed

            start_time = time.time()
            responses = self._generate_responses(model_id, prompt)
            end_time = time.time()
            st.write(f"--- Finished: {model_id} (took {end_time - start_time:.2f}s) ---")


            model_responses[model_id] = responses # Store raw responses

            # Extract recommendations and stories from responses
            temp_recs = []
            temp_stories = []
            for resp_idx, response in enumerate(responses):
                # Pass the original, unprocessed response to extraction
                story, viz_type = self._extract_story_and_viz(response, available_viz_options)
                if viz_type != "Error": # Only add valid recommendations
                    temp_recs.append(viz_type)
                    # Store the cleaned story part
                    temp_stories.append(story if story else "No story generated.")
                else:
                     st.warning(f"Could not extract valid recommendation from response {resp_idx+1} using {model_id}. Response snippet: {response[:200]}...")

            all_recommendations.extend(temp_recs)
            all_stories.extend(temp_stories)


            # Unload model if sequential strategy
            if self.loading_strategy == 'sequential':
                self._unload_model(model_id)

            # Update progress bar after processing each model
            if progress_bar:
                 progress_bar.progress((i + 1) / total_models)


        # Final consolidation message (within status context in app.py)
        st.write("Consolidating results...")

        # Unload all models if concurrent strategy (unload only those loaded in this run)
        if self.loading_strategy == 'concurrent':
            st.write("Unloading concurrent models...")
            models_to_unload = [m for m in loaded_concurrently if m in self.loaded_models]
            for model_id in models_to_unload:
                self._unload_model(model_id)


        # Perform majority voting
        if not all_recommendations:
            st.error("No valid recommendations were generated by any model.")
            # status.update(label="Analysis failed: No recommendations.", state="error") # Update status in app.py
            return "Error: No recommendations generated.", "Table", {}, []

        recommendation_counts = Counter(all_recommendations)
        # Find the most common recommendation(s)
        # Sort items by count descending, then alphabetically for tie-breaking
        sorted_counts = sorted(recommendation_counts.items(), key=lambda item: (-item[1], item[0]))

        if sorted_counts:
             final_recommendation = sorted_counts[0][0]
             max_votes = sorted_counts[0][1]
        else:
             final_recommendation = "Table" # Default if something went wrong
             max_votes = 0


        # Select a representative story (e.g., from the winning recommendation)
        # Find the first story associated with the winning recommendation
        representative_story = "No representative story found."
        if final_recommendation != "Table": # Don't search if default was chosen due to error
             # Find index of first occurrence of the winning recommendation
             try:
                  first_win_index = all_recommendations.index(final_recommendation)
                  if first_win_index < len(all_stories):
                       representative_story = all_stories[first_win_index]
             except ValueError:
                  pass # Should not happen if final_recommendation is from the list

        # Final status update done in app.py after this function returns


        return representative_story, final_recommendation, dict(recommendation_counts), all_stories


    def _extract_story_and_viz(self, response: str, available_viz_options: List[str]) -> Tuple[str, str]:
        """
        Extracts the story (text before the final box) and the boxed visualization type.
        Handles potential variations and cleans the story part.
        """
        # Normalize available options for comparison (lowercase, strip whitespace)
        normalized_options = {opt.strip().lower(): opt for opt in available_viz_options}

        # Regex to find the boxed answer at the VERY END of the string.
        # Allows for optional whitespace between content and box, and inside box.
        # Ensures nothing follows the closing brace '}'.
        # re.DOTALL makes '.' match newlines.
        viz_match = re.search(r'\\boxed{answer:\s*([^}]+?)\s*}\s*$', response, re.IGNORECASE | re.DOTALL)

        viz_type = "Error"
        story = response # Start with the original response

        if viz_match:
            # Extract the content inside the box
            extracted_viz = viz_match.group(1).strip()
            # The story is everything BEFORE the start of the match
            story = response[:viz_match.start()].strip()

            # --- Normalize and validate the extracted visualization type ---
            normalized_extracted = extracted_viz.lower()

            # 1. Exact match (case-insensitive)
            if normalized_extracted in normalized_options:
                 viz_type = normalized_options[normalized_extracted] # Use the original casing
            else:
                 # 2. Partial match (check if extracted is substring or vice-versa)
                 best_match = None
                 # Sort options by length descending to match longer names first (e.g., "Box Plot by Category" before "Box Plot")
                 sorted_norm_options = sorted(normalized_options.keys(), key=len, reverse=True)
                 for norm_opt in sorted_norm_options:
                     if norm_opt in normalized_extracted or normalized_extracted in norm_opt:
                         best_match = normalized_options[norm_opt]
                         break # Take first partial match based on length priority

                 if best_match:
                      viz_type = best_match
                      st.warning(f"Extracted viz '{extracted_viz}' matched partially to '{best_match}'. Using '{best_match}'.")
                 else:
                      # 3. No match found
                      st.warning(f"Extracted viz type '{extracted_viz}' not found in available options: {available_viz_options}. Defaulting to Table.")
                      viz_type = "Table" # Default if no valid match

        else:
             # Fallback: If no boxed answer at the end, try finding the *last* mentioned viz type
             st.warning(f"Could not find '\\boxed{{answer: ...}}' at the end of response. Trying fallback search. Response snippet: {response[-200:]}...")
             found_at = -1
             potential_viz = "Error"
             # Iterate options from longest to shortest
             sorted_options = sorted(available_viz_options, key=len, reverse=True)
             for option in sorted_options:
                 try:
                      # Find all occurrences using word boundaries (\b)
                      indices = [m.start() for m in re.finditer(r'\b' + re.escape(option) + r'\b', story, re.IGNORECASE)]
                      if indices:
                          last_occurrence = indices[-1]
                          if last_occurrence > found_at:
                              found_at = last_occurrence
                              potential_viz = option
                 except re.error: pass # Ignore regex errors

             if potential_viz != "Error":
                 st.warning(f"Using fallback: Found last mentioned option '{potential_viz}' in text.")
                 viz_type = potential_viz
                 # Attempt to clean the story by removing the found fallback term *only if* it looks like a leftover recommendation attempt near the end
                 # This part is heuristic and might remove valid text sometimes.
                 # Check if the found option is within the last ~50 chars and preceded by words like "recommend", "suggest", "use", etc.
                 end_context = story[max(0, found_at - 30):]
                 if found_at > len(story) - 50 and re.search(r'(recommend|suggest|choose|select|visualization is|use|try|plot)\s*:?\s*$', story[:found_at].strip(), re.IGNORECASE):
                      story = story[:found_at].strip()
                      # Further remove the potential preceding trigger phrase
                      story = re.sub(r'(recommend|suggest|choose|select|visualization is|use|try|plot)\s*:?\s*$', '', story, flags=re.IGNORECASE).strip()
                      st.info("Attempted to clean leftover recommendation text before fallback visualization.")

             else:
                 st.warning(f"Fallback failed: Could not find any known viz type in the response. Defaulting to Table.")
                 viz_type = "Table" # Default if nothing found

        # Final story cleanup (remove extra newlines, etc.)
        story = re.sub(r'\n\s*\n', '\n\n', story).strip() # Consolidate multiple blank lines

        return story, viz_type


class DataProcessor:
    # Processes the dataframe using utils and LLMHandler.
    def __init__(self, config_path: str = "config.yaml"):
        self.config = utils.load_config(config_path)
        if not self.config:
             raise ValueError("Failed to load configuration.")
        # Initialize LLMHandler only if vLLM is available
        self.llm_handler = LLMHandler(self.config) if VLLM_AVAILABLE else None

    # Modified to accept progress_bar
    def process_dataframe(self, df: pd.DataFrame, df_name: str, progress_bar: Optional[st.delta_generator.DeltaGenerator] = None) -> Dict[str, Any]:
        # Main processing pipeline for a dataframe.
        results = {
            "dataset_name": df_name,
            "metadata": None,
            "visualization_options": [],
            "recommendation": {
                "story": "LLM analysis disabled or failed.",
                "visualization_type": "Table",
                "votes": {},
                "all_stories": []
            },
            "raw_dataframe": df, # Keep original for potential re-analysis
            "processed_dataframe": None,
            "success": False,
            "error": None
        }

        try:
            st.write("🔄 Preprocessing data...")
            processed_df = utils.preprocess_data(df.copy(), self.config)
            results["processed_dataframe"] = processed_df
            st.write("✅ Data preprocessing complete.")

            st.write("📊 Analyzing dataframe structure...")
            metadata = utils.analyze_dataframe(processed_df, self.config)
            results["metadata"] = metadata
            st.write("✅ Data analysis complete.")

            st.write("🎨 Generating visualization options...")
            viz_options = utils.generate_visualization_options(metadata)
            results["visualization_options"] = viz_options
            st.write(f"✅ Found {len(viz_options)} potential visualization types.")

            if not self.llm_handler:
                 results["error"] = "LLM Handler not available (vLLM/transformers missing?)."
                 st.warning(results["error"])
                 results["success"] = True # Mark as success but without LLM part
                 return results

            if not viz_options or viz_options == ["Table"]:
                 st.warning("Limited data characteristics or only Table option generated. Defaulting to Table visualization.")
                 results["recommendation"]["visualization_type"] = "Table"
                 results["recommendation"]["story"] = "Data analysis suggests a Table is the most appropriate representation due to limited distinct features or relationships."
                 results["success"] = True
                 return results

            st.write("✍️ Creating prompt for LLM...")
            prompt = utils.create_llm_prompt(df_name, metadata, viz_options, self.config)
            # Optionally display prompt for debugging
            # with st.expander("Show LLM Prompt"):
            #     st.text_area("LLM Prompt", prompt, height=400)

            st.write("🧠 Getting recommendations from LLM ensemble...")
            # Pass the progress bar to the LLM handler
            story, viz_type, votes, all_stories = self.llm_handler.get_ensemble_recommendation(prompt, viz_options, progress_bar)
            results["recommendation"]["story"] = story
            results["recommendation"]["visualization_type"] = viz_type
            results["recommendation"]["votes"] = votes
            results["recommendation"]["all_stories"] = all_stories
            st.write(f"🗳️ LLM Ensemble recommended: **{viz_type}**")

            results["success"] = True

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            results["error"] = str(e)
            results["success"] = False
            import traceback
            st.error("Traceback:")
            st.code(traceback.format_exc())


        return results
