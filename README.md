# Indic Transliteration and Normalization

This project performs transliteration and normalization of Indic language text. It converts input text written in Indic scripts into a phonetically equivalent representation in the Latin script. The system also normalizes numbers, dates, units, and currency symbols into human-readable English words.

## Task Description

Given an input sentence in an Indic language, the system generates output with the following transformations:

1. Converts dates to readable formats  
2. Converts numbers to words  
3. Expands measurement units  
4. Replaces currency symbols with their full names  
5. Transliterates the normalized text to Latin script

### Example

**Input:**  
"15/03/1990 को, वैज्ञानिक ने $120 में 500mg यौगिक का एक नमूना खरीदा।"

**Output:**  
"Fifteenth March Nineteen Ninety ko, vaigyanik ne One Hundred and Twenty Dollars mein Five Hundred Milligrams yaugik ka ek namoona kharida."

## Method

1. **Language Detection**  
   The Sarvam-v1 model is used to detect the language of the input text.

2. **Rule-Based Normalization**  
   Using Python's inflect library, dates, numbers, currency, and units are normalized.

3. **Transliteration**  
   The normalized text is transliterated using the IndicXlit engine.

4. **Evaluation**  
   The system is evaluated using Word Error Rate (WER) and CHRF scores.

**WER:** 0.4057  
**CHRF:** 71.46

## Files

This repository includes three notebooks:

1. Finetuning Qwen on Aksharantar dataset using Unsloth with PEFT LoRA and 4-bit quantization
2. Finetuning Sarvam-2B on Aksharantar dataset with LoRA and 4-bit quantization
3. Sentence-level finetuning using Sarvam-2B with 4-bit quantization and LoRA

## Resources

- IndicXlit (https://github.com/AI4Bharat/IndicXlit)  
- Aksharantar Dataset (https://huggingface.co/datasets/ai4bharat/Aksharantar)  
- Sarvam Models (https://huggingface.co/sarvamai)  
- Unsloth Finetuning Framework (https://github.com/unslothai/unsloth)  
- inflect Python library for number to word conversion
