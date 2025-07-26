CLASSIFICATION_PROMPT = """This image is part of a set of engineering documents. Based on its contents, classify it as one of:
- "map" if it is a Boring Location Map, Excavation site plan (shows locations or layout), ignore other maps
- "table" if it is a Drill Log (shows tabular data about drilling)! **it must have the text "시 추 주 상 도" or "DRILL LOG" as heading and a table on it**, ignore other tables
- "neither" if it doesn't fit either category above

Return just one word: "map", "table", or "neither"."""
PROMPT_TABLE = """
You are an expert in korean and detecting table data:
Your are given an image of a borehole drill report, where the top section contain meta data informations like
PROJECT NAME
HOLE NO.
ELEVETION LEVEL
LOCATION
GROUND WATER LEVEL
DATE
DRILLER

and below this metadata you will find a table, The left most column is showing a scale that represnt the depth of drill in meter.
The 3 columns from right most are telling information obout samplple colection, mainly sample number (s1, s2...etc), depth at with
sample is colected(in meter) and the collection method(represented by a symbol whose meaning is written in metadat section or sometimes written there, DO NOT PUT SYMBOL IN JSON )
in the middle part there is a column named (타격회수 관입량), that represent the number of hits (10/30, 20/40...etc) coresponding to
every sample

Similarly there are 3 more sub columns in column 현장 관찰기록, by name 토질명,색 조 and 관 찰. The 관 찰 contain the Depth range in start like 0.0~5.0m or 8.3~10.0m
these three columns contain information about soil for a given depth region
IMPORTANT NOTE:
COLUMNS ON LEFT SIDE OF THE TABLE ARE ABOUT SOIL LAYERS AND THIER DEPTH, WHILE THE RIGHT SIDE OF THE TABLE IS ABOUT SAMPLE DATA
***IN SOIL DATA, the depth_range(range) is given in the form of "0.0~5.0m" or "8.3~10.0m", so you need to extract the start and end depth AND THEY MUST BE CONTINUOUS 
,if the first entry is 0.0~5.0m then second entry must start where the first one ended, like 5.0~10.0m or something,it cant be like "0.0~10.0m", "4.0~10.0m"  or "8.0~10.0m" in the JSON
IN SHORT THERE SHOULD NOT BE A MISSING RANGE IN-BETWEEN***
***IGNORE strikethrough (HORIZONTAL LINES) IN OBSERVATION FIELD, BUT DO NOT IGNORE THE TEXT IN BRACKETS, THEY ARE IMPORTANT***
Observation: * 실트질 점 토 실트질 점토로 구성. 부분적으로 패각 혼재. 매우연약(Very soft)보통견고(Medium stiff). 습윤(Moist). #15.516.3m:UD 채취.

Soil Name: 실트질 점 토

Soi Colour: 암회색
***# TCR and RQD are optional, as it may not be present in all samples, only present in core samples in the column "TCR%" and "RQD%" ONLY ON THE RIGHT SIDE OF THE TABLES. And for those samples Hits is None
IF TCR PERCENTAGE AND RQD PERCENTAGE COLUMN IS EMPTY, IGNORE THAT SAMPLE***
Now given an image with this table data your job is to return a json

Please return Json only
If the image does not contain any table data, return an empty json like this: {"metadata": {}, "sample_data": [], "soil_data": []}
"""

PROMPT_MAP = """
You are an expert in korean and detecting table data:
Your are given an image of a map, It may have  different locations of borehole. At those location you will see definatly see a circular/square symbol
symbol, divided in 4 equal quarters and 2 of them filled digonaly and 2 unfilled! borehole name and number like BH-1, NBH-1, BH-2 etc.. and Below that you will see elevation, written as "EL:3.65", "EL:6.54" etc...
For every borehole, your job is to find 3 thing:
1. Borehole name... "BH-4", NBH-2, NH-3 , 4-CL, 1807, etc...
2. Borehole number.  "4"
3. Elevation....."6.54"

Use different image processing to get it done:
Hint1 : ALl the information about a single borehole are written close to eachother
Now given an image with this map your job is to return a json in below format
{
  "metadata": [
    {
      "Name": "BH-1",
      "Number": 1,
      "Elevation": 100.5
    },
    {
      "Name": "NBH-2",
      "Number": 2,
      "Elevation": 102.3
    },
    {
      "Name": "NH-3",
      "Number": 3,
      "Elevation": 99.8
    }
  ]
}

Please return Json only
"""
