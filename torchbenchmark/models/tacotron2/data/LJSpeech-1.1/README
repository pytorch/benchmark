-----------------------------------------------------------------------------
The LJ Speech Dataset

Version 1.0
July 5, 2017
https://keithito.com/LJ-Speech-Dataset
-----------------------------------------------------------------------------


OVERVIEW

This is a public domain speech dataset consisting of 13,100 short audio clips
of a single speaker reading passages from 7 non-fiction books. A transcription
is provided for each clip. Clips vary in length from 1 to 10 seconds and have
a total length of approximately 24 hours.

The texts were published between 1884 and 1964, and are in the public domain.
The audio was recorded in 2016-17 by the LibriVox project and is also in the
public domain.



FILE FORMAT

Metadata is provided in metadata.csv. This file consists of one record per
line, delimited by the pipe character (0x7c). The fields are:

  1. ID: this is the name of the corresponding .wav file
  2. Transcription: words spoken by the reader (UTF-8)
  3. Normalized Transcription: transcription with numbers, ordinals, and
     monetary units expanded into full words (UTF-8).

Each audio file is a single-channel 16-bit PCM WAV with a sample rate of
22050 Hz.



STATISTICS

Total Clips            13,100
Total Words            225,715
Total Characters       1,308,674
Total Duration         23:55:17
Mean Clip Duration     6.57 sec
Min Clip Duration      1.11 sec
Max Clip Duration      10.10 sec
Mean Words per Clip    17.23
Distinct Words         13,821



MISCELLANEOUS

The audio clips range in length from approximately 1 second to 10 seconds.
They were segmented automatically based on silences in the recording. Clip
boundaries generally align with sentence or clause boundaries, but not always.

The text was matched to the audio manually, and a QA pass was done to ensure
that the text accurately matched the words spoken in the audio.

The original LibriVox recordings were distributed as 128 kbps MP3 files. As a
result, they may contain artifacts introduced by the MP3 encoding.

The following abbreviations appear in the text. They may be expanded as
follows:

     Abbreviation   Expansion
     --------------------------
     Mr.            Mister
     Mrs.           Misess (*)
     Dr.            Doctor
     No.            Number
     St.            Saint
     Co.            Company
     Jr.            Junior
     Maj.           Major
     Gen.           General
     Drs.           Doctors
     Rev.           Reverend
     Lt.            Lieutenant
     Hon.           Honorable
     Sgt.           Sergeant
     Capt.          Captain
     Esq.           Esquire
     Ltd.           Limited
     Col.           Colonel
     Ft.            Fort

     * there's no standard expansion of "Mrs."


19 of the transcriptions contain non-ASCII characters (for example, LJ016-0257
contains "raison d'être").

For more information or to report errors, please email kito@kito.us.



LICENSE

This dataset is in the public domain in the USA (and likely other countries as
well). There are no restrictions on its use. For more information, please see:
https://librivox.org/pages/public-domain.


CHANGELOG

* 1.0 (July 8, 2017):
  Initial release

* 1.1 (Feb 19, 2018):
  Version 1.0 included 30 .wav files with no corresponding annotations in
  metadata.csv. These have been removed in version 1.1. Thanks to Rafael Valle
  for spotting this.


CREDITS

This dataset consists of excerpts from the following works:

* Morris, William, et al. Arts and Crafts Essays. 1893.
* Griffiths, Arthur. The Chronicles of Newgate, Vol. 2. 1884.
* Roosevelt, Franklin D. The Fireside Chats of Franklin Delano Roosevelt.
  1933-42.
* Harland, Marion. Marion Harland's Cookery for Beginners. 1893.
* Rolt-Wheeler, Francis. The Science - History of the Universe, Vol. 5:
  Biology. 1910.
* Banks, Edgar J. The Seven Wonders of the Ancient World. 1916.
* President's Commission on the Assassination of President Kennedy. Report
  of the President's Commission on the Assassination of President Kennedy.
  1964.

Recordings by Linda Johnson. Alignment and annotation by Keith Ito. All text,
audio, and annotations are in the public domain.

There's no requirement to cite this work, but if you'd like to do so, you can
link to: https://keithito.com/LJ-Speech-Dataset

or use the following:
@misc{ljspeech17,
  author       = {Keith Ito},
  title        = {The LJ Speech Dataset},
  howpublished = {\url{https://keithito.com/LJ-Speech-Dataset/}},
  year         = 2017
}
