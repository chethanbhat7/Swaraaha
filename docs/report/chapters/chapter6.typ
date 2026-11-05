#import "../lib.typ": *

// --- Chapter 6: Screenshots ---
#chapter_heading[SCREENSHOTS]

#add_image(align(center, image("/assets/Home Page.png", width: 100%)), caption: [Home page of the web interface])

The home page of the Swaraaha web application, shown in Figure 6.1, has an Analyze Speech Sample panel where users can upload a supported audio file or record speech directly, choose the transcription language, and then start the analysis. A reading-material viewer also sits alongside the input controls so that users can read a standardized passage before recording.

#add_image(align(center, image("/assets/resultpage.png", width: 100%)), caption: [Results page of the web interface])

The web results page, shown in Figure 6.2, shows the recorded waveform with localized dysfluency regions, synchronized playback controls, a timestamped live transcription, and confidence scores for prolongation, block, sound repetition, word repetition, and interjection. Users can upload another sample, review the full analysis, or download a PDF report.

#add_image(align(center, image("/assets/Swaraaha Home Page.png", width: 100%)), caption: [Home page of the desktop application])

The home page of the desktop application, shown in Figure 6.3, allows users to open a standardized reading passage, load an audio file from the local file system, or record new speech directly. The interface also shows the current processing status and gives the controls needed to continue once an audio sample has been selected.
The desktop client closely follows the web workflow but runs entirely on the local machine, so it still works where a network connection is unreliable or where the audio cannot leave the clinic. The reading passage viewer sits right in the same window as the input controls, so a speaker can read the standardized text and start recording without switching applications, which reduces the variability that different reading setups introduce. The status indicator reports each processing stage in turn, and the controls stay disabled until a valid audio sample is available, which prevents accidental submissions and keeps the interaction predictable for non-technical users. Because the model package is imported directly instead of called over HTTP, the desktop application avoids the overhead of serializing audio and results, and it still works when the web backend is not running.

#add_image(align(center, image("/assets/Swaraaha Analysis Result Page.png", width: 100%)), caption: [Analysis result page of the desktop application])

The desktop analysis results page, shown in Figure 6.4, has a classification table that reports whether each of the five dysfluency types was detected and the corresponding confidence value, while the localization timeline marks detected events on the waveform. The user can review the analysis, return to the home page, or export the results as a PDF report.
The results page keeps the classification table and the timeline visually linked: selecting a row highlights the corresponding region on the waveform. This link lets a clinician move from a summary judgement to the exact evidence behind it without manually searching the audio, which is the main practical gain from combining classification and localization in one view. The confidence values appear per class instead of being collapsed into one single overall score, so a weak detection stays visible rather than hidden. Exporting the PDF captures the table, waveform overlays, and transcript in one document.
Together, the web and desktop screens show the same pipeline at different levels of deployment.
The web interface favours shared access and report download, while the desktop interface keeps audio and inference local.
In both cases, the visible result connects the class-level prediction, the time-localized event, and the transcript context, so the output can still be reviewed instead of treated as an unexplained label.

#pagebreak()