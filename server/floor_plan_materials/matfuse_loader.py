# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy MatFuse backend — defers material_generator import until a function is called."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _material_generator_module():
    from floor_plan_materials import material_generator

    return material_generator


def material_generate_from_prompt(prompts):
    return _material_generator_module().material_generate_from_prompt(prompts)


def generate_texture_map_from_prompt(prompt):
    return _material_generator_module().generate_texture_map_from_prompt(prompt)


def generate_texture_map_from_prompt_and_sketch(prompt, sketch):
    return _material_generator_module().generate_texture_map_from_prompt_and_sketch(
        prompt, sketch
    )


def generate_texture_map_from_prompt_and_sketch_and_image(prompt, sketch, image):
    return _material_generator_module().generate_texture_map_from_prompt_and_sketch_and_image(
        prompt, sketch, image
    )


def generate_texture_map_from_prompt_and_color(prompt, color):
    return _material_generator_module().generate_texture_map_from_prompt_and_color(
        prompt, color
    )


def generate_texture_map_from_prompt_and_color_and_sketch(prompt, color, sketch):
    return _material_generator_module().generate_texture_map_from_prompt_and_color_and_sketch(
        prompt, color, sketch
    )


def generate_texture_map_from_prompt_and_color_palette(prompt, color_palette):
    return _material_generator_module().generate_texture_map_from_prompt_and_color_palette(
        prompt, color_palette
    )
