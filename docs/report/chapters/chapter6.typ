#import "../lib.typ": *

// --- Chapter 6: Screenshots ---
#chapter_heading[SCREENSHOTS]

#add_image(align(center, image("/assets/Home Page.png", width: 100%)), caption: [Home page of the web interface])

The home page of the Swaraaha web application, shown in Figure 6.1, features the Analyze Speech Sample panel, which allows users to upload a supported audio file or record speech directly, select the transcription language, and begin the analysis. A reading-material viewer is provided alongside the input controls so that users can read a standardized passage before recording.

#add_image(align(center, image("/assets/resultpage.png", width: 100%)), caption: [Results page of the web interface])

The web results page, shown in Figure 6.2, presents the recorded waveform with localized dysfluency regions, synchronized playback controls, a timestamped live transcription, and confidence scores for prolongation, block, sound repetition, word repetition, and interjection. Users can upload another sample, review the complete analysis, or download a PDF report.

#add_image(align(center, image("/assets/Swaraaha Home Page.png", width: 100%)), caption: [Home page of the desktop application])

The home page of the desktop application, shown in Figure 6.3, allows users to open a standardized reading passage, load an audio file from the local file system, or record new speech. The interface also displays the current processing status and provides the controls required to continue once an audio sample has been selected.
The desktop client mirrors the web workflow but runs entirely on the local machine, so it can be used in settings where a network connection is unreliable or where audio cannot leave the clinic. The reading passage viewer is embedded in the same window as the input controls, which means a speaker can read the standardized text and start recording without switching applications, reducing the variability introduced by different reading setups. The status indicator reports each processing stage in sequence, and the controls remain disabled until a valid audio sample is available, which prevents accidental submissions and makes the interaction predictable for non-technical users. Because the model package is imported directly rather than called over HTTP, the desktop application avoids the overhead of serializing audio and results, and it remains usable when the web backend is not running.

#add_image(align(center, image("/assets/Swaraaha Analysis Result Page.png", width: 100%)), caption: [Analysis result page of the desktop application])

The desktop analysis results page, shown in Figure 6.4, presents a classification table that reports whether each of the five dysfluency types was detected and the corresponding confidence value, while the localization timeline marks detected events on the waveform. The user can review the analysis, return to the home page, or export the results as a PDF report.
The results page keeps the classification table and the timeline visually linked: selecting a row highlights the corresponding region on the waveform. This coupling lets a clinician move from a summary judgement to the exact evidence behind it without manually searching the audio, which is the main practical advantage of combining classification and localization in one view. The confidence values are shown per class rather than collapsed into a single score, so a weak detection is visible instead of being hidden. Exporting the PDF captures the table, waveform overlays, and transcript in one document.

#pagebreak()