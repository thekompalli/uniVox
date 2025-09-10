# PyAnnote Model Access Instructions

## Overview
PyAnnote models (segmentation-3.0 and speaker-diarization-3.1) are **gated repositories** on HuggingFace. Your current HuggingFace token alone is not sufficient - you need to complete additional steps to gain access.

## Required Steps

### 1. Accept User Conditions for Both Models
You must visit these URLs and accept the user conditions:

- **Segmentation Model**: https://huggingface.co/pyannote/segmentation-3.0
- **Diarization Model**: https://huggingface.co/pyannote/speaker-diarization-3.1

### 2. Fill Required Information
On each model page, you'll need to provide:
- **Company/University**: Your organization name
- **Website**: Your organization's website or personal website
- **Brief description**: How you plan to use the models

### 3. Wait for Approval
- Access is typically granted automatically after filling the forms
- You should receive confirmation emails once approved
- This process usually takes a few minutes to a few hours

## Why Are These Models Gated?

The models are gated to:
- Help maintainers understand the user base
- Improve the models based on usage feedback
- Occasionally notify users about premium features and paid services
- The models remain open-source under MIT license

## Current Status

✅ **Fixed Issues:**
- WeSpeaker repository path corrected to: `pyannote/wespeaker-voxceleb-resnet34-LM`
- Download scripts updated with correct model paths

❌ **Pending Access:**
- PyAnnote segmentation-3.0: Requires gated repository access
- PyAnnote speaker-diarization-3.1: Requires gated repository access

## Next Steps

1. Complete the access requests for both PyAnnote models
2. Once approved, the download script will work with your existing HuggingFace token
3. All other models should download successfully with your current token

## Alternative Approach

If you prefer not to request gated access, you can:
- Use older versions of PyAnnote models that may be publicly available
- Use alternative speaker diarization libraries
- Continue development with mock models until access is granted