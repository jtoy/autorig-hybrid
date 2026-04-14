    Summary
## Challenge

  We need a developer to improve the character-part cutting pipeline in this repo.

  ## Goal

  Update the code so that when we run the batch cutter for all characters, the output parts are production-usable.

Here are 2 videos explaining it:

https://orchestrator.distark.com/dropbox/MPNfb7/file/vWs3KE?inline=true

https://orchestrator.distark.com/dropbox/MPNfb7/file/Wms7rd?inline=true

here is the repo: 
https://github.com/jtoy/autorig-hybrid

  ## What must be fixed

  ### 1. Better cuts
  The AI-generated cuts must be correct and clean.

  That includes:
  - cutting each body part accurately
  - preserving silhouettes and edges
  - inferring hidden or occluded areas when one part is behind another part
  - completing missing portions of a part when the source image does not fully show it because of overlap

  Example:
  If an arm is partially behind the torso, the output for that arm should still be a complete usable part, not just the visible fragment.

  ### 2. Correct part size
  Each exported part must keep the correct scale relative to the original image.

  That means:
  - the generated or refined part should not be enlarged or shrunk incorrectly
  - the final PNG for each part should match the original image scale
  - proportions must stay consistent with the source character

  ## Current workflow

  The current pipeline is driven by:

  ```bash
  python lasso_batch.py character

  Example:

  time python lasso_batch.py tank

  It uses saved lasso polygons, creates raw part crops, optionally refines them with an AI model, removes background, and writes PNGs to outputs/character/.

  ## Deliverable

  We want code changes that make the pipeline reliably produce:

  - better body-part separation
  - believable completion of occluded or hidden areas
  - output parts at the same scale as the source image

  ## Required submission

  You must run the pipeline for all characters in resources / the existing character set used by this repo, not just one sample.

  Submit:

  - the code changes
  - the generated outputs for all characters
  - a short note explaining what you changed and why

  ## How we will evaluate

  We will run the pipeline ourselves for multiple characters, including:

  time python lasso_batch.py tank

  We expect you to have already run it across the full available character set and submitted those results to us.

  Your work is successful if:

  - the cuts look visually correct
  - hidden parts are inferred well when needed
  - the output parts are complete and usable for rigging
  - the part sizes match the original character scale
  - results are consistently good across all characters in the repo
