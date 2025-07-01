import json
import os
from pathlib import Path
from typing import Optional

import PIL.Image

# from utils.format2bio import BIOFormatter
from tqdm import tqdm


def save_bio(
    input_file_path: str,
    output_file_path: str,
    entities: list = ["title", "section", "author"],
):
    formatter = BIOFormatter(
        input_file_path,
        output_file_path,
    )
    formatter.process_data(entities=entities)


def get_data(
    input_folder_path: str,
    output_folder_path: Optional[str] = None,
    run_bio_preprocessing: bool = False,
    return_images: bool = False,
):
    """

    Args:
        input_folder_path (str): path to input folder with doc_bank jsons
        output_folder_path (Optional[str], optional): output folder to save jsons with BIO notation. Defaults to None.
        run_bio_preprocessing (bool, optional): Wether to reprocess jsons or not. Defaults to False.
        return_images (bool, optional): Reads image files into memories from input_folder_path. Defaults to False.

    Returns:
        Union[list, Tuple[list, list]]: returns list of dictionaries (parsed jsons) and optionally list of read images
    """
    if run_bio_preprocessing:
        assert (
            output_folder_path is not None
        ), "output_folder_path should be set so save bio processing results"

    data = []
    doc_images = []
    folders_list = [p[0] for p in os.walk(input_folder_path)][1:]
    for folder in tqdm(folders_list, desc="Loading Source Files"):
        if ".ipynb_checkpoints" not in folder:
            for path in Path(folder).rglob("*.json"):
                if run_bio_preprocessing:
                    # puts files under output_folder_path/train/doc_id/doc.json and the same for test
                    out_path = Path(output_folder_path) / Path(*path.parts[-3:])
                    save_bio(path, out_path)
                    with open(out_path) as f:
                        data_file = json.load(f)
                else:
                    with open(path) as f:
                        data_file = json.load(f)
                data.append(data_file)
                if return_images:
                    image_path = str(path)[:-5] + "_ori.jpg"
                    img = PIL.Image.open(image_path)
                    doc_image = img.copy()
                    img.close()
                    doc_images.append(doc_image)

    return (data, doc_images) if return_images else data
