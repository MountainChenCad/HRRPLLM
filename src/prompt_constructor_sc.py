# src/prompt_constructor_sc.py
import random

try:
    from config import TARGET_HRRP_LENGTH as FALLBACK_TARGET_HRRP_LENGTH
except ImportError:
    FALLBACK_TARGET_HRRP_LENGTH = 1000
    print("Warning (PromptConstructorSC): Could not import FALLBACK_TARGET_HRRP_LENGTH from config, using default 1000.")


class PromptConstructorSC:
    def __init__(self, dataset_name_key, class_names_for_task, sc_encoding_config,
                 # --- Ablation Flags ---
                 include_system_instruction=True,
                 include_background_knowledge=True,
                 include_candidate_list=True,
                 include_output_format_instruction=True
                 ):
        self.dataset_name_key = dataset_name_key
        self.class_names_for_task = class_names_for_task
        self.sc_encoding_config = sc_encoding_config
        self.hrrp_length_info = self.sc_encoding_config.get('TARGET_HRRP_LENGTH_INFO', FALLBACK_TARGET_HRRP_LENGTH)

        self.include_system_instruction = include_system_instruction
        self.include_background_knowledge = include_background_knowledge
        self.include_candidate_list = include_candidate_list
        self.include_output_format_instruction = include_output_format_instruction

        self.context_header = self._build_context_header_for_sc()

    def _build_context_header_for_sc(self):
        # --- English Translations ---
        task_definition = (
            "You are a radar target recognition expert, skilled at identifying target types by analyzing their scattering center characteristics."
            "Your task is to accurately identify the target from a list of candidates based on the provided primary scattering center information (position index and relative amplitude)."
        )

        sc_description = (
            "Scattering centers are the primary regions on a target where radar echo energy is concentrated. They typically correspond to geometric discontinuities, edges, corners, or strong reflective surfaces of the target."
            "By analyzing the number of scattering centers, their relative positions (range bin indices), and their respective relative amplitudes, one can infer the target's size, shape, and structural features."
            f"In this task, the provided scattering center information is extracted from a 1D High-Resolution Range Profile (HRRP) of length {self.hrrp_length_info} and is sorted in descending order of amplitude."
            "The 'position index' starts counting from 0, representing the position in the original HRRP sequence."
            "The 'relative amplitude' is normalized (e.g., the maximum value is 1)."
        )

        if "simulated" in self.dataset_name_key.lower():
            dataset_info_prefix = "The data currently being analyzed originates from **simulated HRRP scattering center data**."
        elif "measured" in self.dataset_name_key.lower():
            dataset_info_prefix = "The data currently being analyzed originates from **measured HRRP scattering center data**."
        else:
            dataset_info_prefix = "The data currently being analyzed is HRRP scattering center data."

        candidate_list_text = ""
        if self.include_candidate_list and self.class_names_for_task:
            candidate_list_text = f" Candidate target classes include: `{', '.join(self.class_names_for_task)}`."
        elif not self.class_names_for_task and self.include_candidate_list:
            print(
                "Warning (PromptConstructorSC): include_candidate_list is True, but class_names_for_task is empty. No candidate list will be included.")

        dataset_info = dataset_info_prefix + candidate_list_text

        reasoning_guidance_intro = "**Reasoning Steps and Requirements:**\n"
        reasoning_step1 = (
            "1.  **Examine Test Sample Scattering Centers**: Carefully observe the data provided in the 'Test Sample Scattering Centers' section. Focus on:\n"
            "    *   The number of detected scattering centers.\n"
            "    *   The position indices and relative amplitudes of the strongest few scattering centers.\n"
            f"    *   The approximate distribution pattern of scattering centers across the entire target length (0 to {self.hrrp_length_info - 1}) (e.g., concentrated at the front, rear, evenly distributed, etc.).\n"
        )
        reasoning_step2 = "2.  **Reference Neighboring/Support Samples (if provided)**: Compare the scattering center features of the test sample with those of known class samples in the 'Neighboring Training Sample Reference'.\n" \
                          "    *   Note the known class of each reference sample and compare the similarity of its scattering center pattern to the test sample.\n"
        reasoning_step3 = "3.  **Make a Comprehensive Judgment**: Based on your understanding of scattering center distribution patterns for different target types and the comparison with reference samples, determine which candidate class the test sample most closely matches.\n"

        output_format_instruction_text = (
            "4.  **Output Format**:\n"
            "    *   On the first line of your response, please clearly state the predicted target class in the format: `Predicted Target Class: [Fill in one of the candidate class names here]`\n"
            "    *   In subsequent lines, briefly state your main reasons for this judgment, e.g., based on the number, position, or specific pattern of scattering centers, or similarity to a reference sample."
        )
        # --- End English Translations ---

        reasoning_parts = []
        if self.include_system_instruction:
            reasoning_parts.append(reasoning_step1)
            reasoning_parts.append(reasoning_step2)
            reasoning_parts.append(reasoning_step3)

        if self.include_output_format_instruction:
            if not self.include_system_instruction and reasoning_parts:
                output_format_instruction_text = output_format_instruction_text.replace("4.", f"{len(reasoning_parts) + 1}.")
            elif not self.include_system_instruction:
                output_format_instruction_text = output_format_instruction_text.replace("4.", "1.")
            reasoning_parts.append(output_format_instruction_text)

        full_reasoning_guidance = ""
        if reasoning_parts:
            full_reasoning_guidance = reasoning_guidance_intro + "".join(reasoning_parts)

        header_parts = []
        if self.include_system_instruction:
            header_parts.append(f"{task_definition}\n\n")
        if self.include_background_knowledge:
            header_parts.append(f"**Scattering Center Characteristics Overview:**\n{sc_description}\n\n")
        header_parts.append(f"**Current Dataset and Task:**\n{dataset_info}\n\n")
        if full_reasoning_guidance:
            header_parts.append(f"{full_reasoning_guidance}\n\n")
        header_parts.append(f"------------------------------------\n")
        return "".join(header_parts)

    def construct_prompt_with_sc(self, query_sc_text, neighbor_sc_examples=None):
        prompt = self.context_header

        if neighbor_sc_examples and len(neighbor_sc_examples) > 0:
            prompt += "**Neighboring Training Sample Reference (Support Set Prototypes):**\n" # English
            for i, (neighbor_text, neighbor_label) in enumerate(neighbor_sc_examples):
                prompt += f"\n--- Reference Prototype {i + 1} ---\n" # English
                prompt += f"Known Target Class: `{neighbor_label}`\n" # English
                prompt += f"Its primary scattering center information:\n{neighbor_text}\n" # English
            prompt += "------------------------------------\n"
        elif self.include_system_instruction:
            prompt += "**Note: No neighboring training samples (prototypes) are provided for this prediction (0-shot task). Please make your judgment based on the scattering center characteristics overview and your own knowledge.**\n" # English
            prompt += "------------------------------------\n"

        prompt += "**Test Sample Scattering Centers (Please predict based on this):**\n" # English
        prompt += f"{query_sc_text}\n\n"

        if self.include_output_format_instruction:
            prompt += "Please strictly follow the output format requirements.\n" # English
            prompt += "Predicted Target Class: " # English
        elif self.include_candidate_list:
            prompt += "Predicted Target Class: " # English
        else:
            prompt += "Your judgment is: " # English

        return prompt


if __name__ == "__main__":
    try:
        from config import SCATTERING_CENTER_ENCODING as mock_sc_encoding_config_main
        from config import TARGET_HRRP_LENGTH as mock_target_hrrp_length_main

        if 'TARGET_HRRP_LENGTH_INFO' not in mock_sc_encoding_config_main:
            mock_sc_encoding_config_main['TARGET_HRRP_LENGTH_INFO'] = mock_target_hrrp_length_main
    except ImportError:
        mock_target_hrrp_length_main = 1000
        mock_sc_encoding_config_main = {
            "format": "list_of_dicts", "precision_pos": 0, "precision_amp": 3,
            "TARGET_HRRP_LENGTH_INFO": mock_target_hrrp_length_main
        }
        print("Warning (PromptConstructorSC): Could not import from config, using default test configurations.")

    # Assuming scattering_center_encoder.py is in the same directory or PYTHONPATH
    try:
        from scattering_center_encoder import encode_single_sc_set_to_text
    except ImportError:
        print("Error: Could not import encode_single_sc_set_to_text. Make sure scattering_center_encoder.py is accessible.")
        # Define a dummy function if not available for the test to run without erroring out immediately
        def encode_single_sc_set_to_text(sc_set, encoding_config):
            return "Dummy SC Text"


    mock_dataset_name_sc = "simulated_sc_test"
    mock_class_names_for_task_sc = ["F-22", "T-72", "MQ-9"]

    dummy_sc_1 = [(100, 0.9), (150, 0.7), (50, 0.6)]
    query_sc = [(98, 0.88), (152, 0.72), (45, 0.65)]
    sc_text_1 = encode_single_sc_set_to_text(dummy_sc_1, mock_sc_encoding_config_main)
    query_sc_text_main = encode_single_sc_set_to_text(query_sc, mock_sc_encoding_config_main)
    mock_neighbors_sc = [(sc_text_1, "F-22")]

    print("\n--- Testing Full Prompt (English) ---")
    constructor_full = PromptConstructorSC(mock_dataset_name_sc, mock_class_names_for_task_sc,
                                           mock_sc_encoding_config_main)
    prompt_full = constructor_full.construct_prompt_with_sc(query_sc_text_main, mock_neighbors_sc)
    print(prompt_full[:600] + "...") # Print a bit more for English
    with open("test_prompt_ablation_full_en.txt", "w", encoding="utf-8") as f:
        f.write(prompt_full)

    print("\n--- Testing Ablation: No System Instruction (English) ---")
    constructor_no_sys = PromptConstructorSC(mock_dataset_name_sc, mock_class_names_for_task_sc,
                                             mock_sc_encoding_config_main, include_system_instruction=False)
    prompt_no_sys = constructor_no_sys.construct_prompt_with_sc(query_sc_text_main, mock_neighbors_sc)
    print(prompt_no_sys[:600] + "...")
    with open("test_prompt_ablation_no_sys_en.txt", "w", encoding="utf-8") as f:
        f.write(prompt_no_sys)

    # ... (you can add more ablation tests here for English prompts if needed) ...

    print("\n--- Testing Ablation: 0-shot (English) ---")
    prompt_0shot = constructor_full.construct_prompt_with_sc(query_sc_text_main, None)
    print(prompt_0shot[:600] + "...")
    with open("test_prompt_ablation_0shot_en.txt", "w", encoding="utf-8") as f:
        f.write(prompt_0shot)