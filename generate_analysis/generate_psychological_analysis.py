#!/usr/bin/env python3
"""
Generate the MLLM psychological analysis (and confound-control conditions) for
the Intentonomy dataset with Janus-Pro-7B.

The psychological analysis is the text feature used by PsyIntent. It is produced
offline, once per image, and cached; it does not enter the per-query deployment
path. The prompt template is derived (with assistance from GPT-4o) from
established findings on social intent: emotion and intent are mutually dependent
in social contexts (Mittal et al., 2024), and cultural and bodily cues are key
predictors of intent (Jia et al., 2021; Sartori et al., 2011). It is streamlined
to a single sentence so the output stays grounded in visual evidence.

Prompt types (--prompt_type):
  psy            The psychological analysis used by PsyIntent (default).
  generic        A plain, non-psychological description of the image.
  objscene       A literal object/scene list (no emotion or intent).
  shuffled       The genuine psychological texts permuted across images
                 (destroys image-text correspondence). No GPU needed.
  random         A fixed meaningless placeholder text. No GPU needed.

The generic / objscene / shuffled / random conditions reproduce the
confound-control experiment: they feed the full PsyIntent architecture different
MLLM text conditions, isolating the contribution of the psychological analysis.

Janus-Pro-7B inference (Sec. 4.1.3): nucleus sampling, temperature = 1.0,
top-p = 0.95, max_new_tokens = 256.

Usage:
  cd PsyIntent/generate_analysis
  python generate_psychological_analysis.py --prompt_type psy --splits train val test \
      --model_path /path/to/Janus-Pro-7B \
      --image_root /path/to/intentonomy/low \
      --anno_dir  /path/to/annotations/intentonomy \
      --output_dir /path/to/annotations/intentonomy

Outputs are written as {split}_janus7b_{suffix}.json (psy -> "_psy"), preserving
the original COCO-style annotation structure with the field "caption_by_Janus_7B".
"""
import os
import sys
import json
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# ============================================================
# Prompt templates
# ============================================================
PROMPTS = {
    "psy": (
        "Briefly analyze the image within a sentence to identify psychological cues "
        "about the emotions, intentions, and social context. Focus on visual details—like "
        "expressions, body language, and setting—that hint at their mood, self-perception, "
        "and possible motivations."
    ),
    "generic": (
        "Briefly describe what is happening in this image in one or two sentences. "
        "Focus on the visible content, actions, and setting."
    ),
    "objscene": (
        "Generate a comma-separated list of succinct object descriptions, visual details, "
        "or stylistic elements visible in this image, ordered from the most to the least significant. "
        "Do not interpret emotions or intentions—only describe what is literally visible."
    ),
}

OUTPUT_SUFFIX = {
    "psy": "psy",
    "generic": "generic",
    "objscene": "objscene",
    "shuffled": "shuffled",
    "random": "random",
}

# Caption field name used by the data loader.
CAPTION_FIELD = "caption_by_Janus_7B"

# Source annotation files (the psychological condition is read from these so the
# shuffled condition can reuse the genuine texts).
SPLIT_FILES = {
    "train": "train_janus7b_psy.json",
    "val":   "val_janus7b_psy.json",
    "test":  "test_janus7b_psy.json",
}

# Janus-Pro-7B sampling configuration (Sec. 4.1.3).
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 256

lock = threading.Lock()


# ============================================================
# Janus-Pro model
# ============================================================
def load_janus_model(model_path):
    """Load Janus-Pro-7B once."""
    import torch
    from transformers import AutoModelForCausalLM
    from janus.models import MultiModalityCausalLM, VLChatProcessor

    print(f"Loading Janus-Pro-7B from {model_path} ...")
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    ).to(torch.bfloat16).cuda().eval()
    print("Janus-Pro-7B loaded.")
    return vl_chat_processor, tokenizer, vl_gpt


def generate_single(item, image_root, prompt, vl_chat_processor, tokenizer, vl_gpt):
    """Generate the analysis for a single image with Janus-Pro-7B."""
    from janus.utils.io import load_pil_images

    image_path = os.path.join(image_root, item['image_id'] + '.jpg')
    if not os.path.exists(image_path):
        return item, ""

    try:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        with lock:
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                use_cache=True,
            )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    except Exception as e:
        print(f"\nError processing {item['image_id']}: {e}")
        answer = ""

    return item, answer


# ============================================================
# Batch generation (MLLM conditions)
# ============================================================
def generate_mllm_captions(splits, prompt_type, num_threads, model_components,
                           anno_dir, output_dir):
    """Generate captions for multiple splits with Janus-Pro-7B (model loaded once)."""
    vl_chat_processor, tokenizer, vl_gpt = model_components
    prompt = PROMPTS[prompt_type]
    suffix = OUTPUT_SUFFIX[prompt_type]

    for split in splits:
        anno_file = os.path.join(anno_dir, SPLIT_FILES[split])
        with open(anno_file, 'r') as f:
            data = json.load(f)

        total = len(data['annotations'])
        updated = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(
                    generate_single, item, None, prompt,
                    vl_chat_processor, tokenizer, vl_gpt
                ): item for item in data['annotations']
            }
            with tqdm(total=total, desc=f"[{split}] {prompt_type}", unit="img") as pbar:
                for future in as_completed(futures):
                    item, answer = future.result()
                    item[CAPTION_FIELD] = answer
                    updated.append(item)
                    pbar.update(1)

        # keep stable ordering by id
        updated.sort(key=lambda x: x.get('id', 0))

        out_file = os.path.join(output_dir, f"{split}_janus7b_{suffix}.json")
        data['annotations'] = updated
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(updated)} captions -> {out_file}")


# ============================================================
# Shuffled / Random (no GPU)
# ============================================================
def generate_shuffled_psy(splits, anno_dir, output_dir):
    """Permute the genuine psychological texts across images (no GPU)."""
    for split in splits:
        anno_file = os.path.join(anno_dir, SPLIT_FILES[split])
        with open(anno_file, 'r') as f:
            data = json.load(f)

        all_texts = [item.get(CAPTION_FIELD, '') for item in data['annotations']]
        random.shuffle(all_texts)
        for i, item in enumerate(data['annotations']):
            item[CAPTION_FIELD] = all_texts[i]

        out_file = os.path.join(output_dir, f"{split}_janus7b_shuffled.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data['annotations'])} shuffled captions -> {out_file}")


def generate_random_text(splits, anno_dir, output_dir):
    """Fill a fixed meaningless placeholder text (no GPU)."""
    placeholder = "The image shows various elements and objects in a scene with colors and shapes."
    for split in splits:
        anno_file = os.path.join(anno_dir, SPLIT_FILES[split])
        with open(anno_file, 'r') as f:
            data = json.load(f)

        for item in data['annotations']:
            item[CAPTION_FIELD] = placeholder

        out_file = os.path.join(output_dir, f"{split}_janus7b_random.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data['annotations'])} random captions -> {out_file}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate psychological analysis (Janus-Pro-7B)")
    parser.add_argument('--prompt_type', type=str, default='psy',
                        choices=['psy', 'generic', 'objscene', 'shuffled', 'random'],
                        help="Type of analysis to generate (default: psy)")
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help="Dataset splits to process")
    parser.add_argument('--num_threads', type=int, default=4,
                        help="Number of threads for MLLM inference")
    parser.add_argument('--model_path', type=str, required=False,
                        default='./models/Janus-Pro-7B',
                        help="Path to the Janus-Pro-7B weights")
    parser.add_argument('--image_root', type=str, required=False,
                        default='../_data/images/intentonomy/low',
                        help="Directory containing Intentonomy images ({image_id}.jpg)")
    parser.add_argument('--anno_dir', type=str, required=False,
                        default='../_data/annotations/intentonomy',
                        help="Directory with the source *_janus7b_psy.json annotation files")
    parser.add_argument('--output_dir', type=str, required=False, default=None,
                        help="Where to write the generated JSONs (default: same as --anno_dir)")
    parser.add_argument('--seed', type=int, default=666, help="Random seed for the shuffled condition")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = args.output_dir or args.anno_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Image root : {os.path.abspath(args.image_root)}")
    print(f"Anno dir   : {os.path.abspath(args.anno_dir)}")
    print(f"Output dir : {os.path.abspath(output_dir)}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Splits     : {args.splits}")
    print(f"Sampling   : nucleus sampling, temperature={TEMPERATURE}, top_p={TOP_P}, "
          f"max_new_tokens={MAX_NEW_TOKENS}")

    if args.prompt_type in ('psy', 'generic', 'objscene'):
        assert os.path.isdir(args.model_path), f"Janus-Pro-7B model not found at {args.model_path}"
        model_components = load_janus_model(args.model_path)
        generate_mllm_captions(args.splits, args.prompt_type, args.num_threads,
                               model_components, args.anno_dir, output_dir)
    elif args.prompt_type == 'shuffled':
        generate_shuffled_psy(args.splits, args.anno_dir, output_dir)
    elif args.prompt_type == 'random':
        generate_random_text(args.splits, args.anno_dir, output_dir)


if __name__ == "__main__":
    main()
