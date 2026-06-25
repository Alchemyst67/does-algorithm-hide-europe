# Final Findings Brief

This file is generated from the executed final notebook outputs. It is intended as a quick oral-presentation briefing, not as a replacement for the notebook.

## Does the algorithm hide Europe?

Model-dependent. Worst Europe PACPG@20 is Popularity (-10.3%); highest Europe exposure is CLIP-image-content (32.8%).

Evidence section: 11, 12. Confidence: moderate. Caveat: Offline sampled audit with proxy metadata..

## Does the recommender show local Europe or globally compatible Europe?

Platform-compatible Europe has mean Exposure@20=18.1% and mean PACPG@20=-1.4%; Local Europe has mean Exposure@20=0.6% and mean PACPG@20=-0.4%; the stricter local non-English/no-US subset has mean Exposure@20=0.8%.

Evidence section: 21. Confidence: exploratory/moderate. Caveat: DNA scores are transparent proxies; TMDb provider data and LUMIERE admissions are documented next layers, not included as fake data..

## Which European countries are least visible?

France has the lowest mean PACPG@20 among support-passing European countries (-1.7%).

Evidence section: 12. Confidence: moderate. Caveat: Low support limits claims..

## Which languages are most visible?

English has the highest mean language Exposure@20 (97.8%).

Evidence section: 13. Confidence: moderate. Caveat: Original-language metadata is a proxy..
