import io
import os  # noqa F401
from concurrent.futures import ThreadPoolExecutor

import fitz
import torch  # noqa F401
from deepseek_ocr import DeepseekOCRForCausalLM
from PIL import Image
from process.image_process import DeepseekOCRProcessor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def pdf_to_images_high_quality(pdf_bytes, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []

    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(
                    img, mask=img.split()[-1] if img.mode == "RGBA" else None
                )
                img = background

        images.append(img)

    pdf_document.close()
    return images


def process_single_image(image, prompt, crop_mode):
    """single image"""
    prompt_in = prompt
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=crop_mode
            )
        },
    }
    return cache_item


def pdf_to_text(
    input_pdf_bytes,
    model_path,
    prompt,
    crop_mode,
    max_concurrency,
    num_workers,
    skip_repeat,
):
    ModelRegistry.register_model(
        "DeepseekOCRForCausalLM", DeepseekOCRForCausalLM
    )  # noqa: E501  # noqa: E501

    llm = LLM(
        model=model_path,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=max_concurrency,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True,
    )

    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
        )
    ]  # window for fast；whitelist_token_ids: <td>,</td>

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    print(f"{Colors.RED}PDF loading .....{Colors.RESET}")

    images = pdf_to_images_high_quality(input_pdf_bytes)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_inputs = list(
            tqdm(
                executor.map(
                    process_single_image,
                    images,
                    [prompt] * len(images),
                    [crop_mode] * len(images),
                ),
                total=len(images),
                desc="Pre-processed images",
            )
        )

    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

    pages = {}
    jdx = 0
    for output in outputs_list:
        content = output.outputs[0].text

        if "<｜end▁of▁sentence｜>" in content:  # repeat no eos
            content = content.replace("<｜end▁of▁sentence｜>", "")
        else:
            if skip_repeat:
                continue

        pages[f"page_{jdx}"] = {
            "text": content,
        }

        jdx += 1

    return pages
