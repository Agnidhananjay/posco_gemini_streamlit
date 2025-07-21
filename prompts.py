CLASSIFICATION_PROMPT = """This image is part of a set of engineering documents. Based on its contents, classify it as one of:
- "map" if it is a Boring Location Map, Excavation site plan (shows locations or layout), ignore other maps
- "table" if it is a Drill Log (shows tabular data about drilling), ignore other tables
- "neither" if it doesn't fit either category above

Return just one word: "map", "table", or "neither"."""
PROMPT_TABLE = """
You are an expert in korean and detecting table data:
Your are given an image of a borehole drill report, where the top section contain meta data informations like
PROJECT NAME
HOLE NO.
ELEV
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

Now given an image with this table data your job is to return a json in below format
{
  metadata:{
    'PROJECT NAME':------,
    'HOLE NO.':---------,
    'ELEV':-----------,
    'LOCATION':--------,
    'GROUND WATER LEVEL':------,
    'DATE':---------
    'DRILLER':---------
  },
  sample_data:[
         {
          'sample_number':----,
          'Depth':----,
          'Hits':----,
          'Method':----
    },
        {
          'sample_number':----,
          'Depth':----,
          'Hits':----,
          'Method':----
    },

   ]
   soil_data:[
                    {
                    'depth_range':-----
                    'soil_name':----,
                    'soil_color':----,
                    'observation':----

                    },
                    {
                    'depth_range':-----
                    'soil_name':----,
                    'soil_color':----,
                    'observation':----

                    },

   ]
}
Please return Json only
"""

PROMPT_MAP = """
You are an expert in korean and detecting table data:
Your are given an image of a map, that has different locations of borehole. At those location you will see three thing a circular
symbol filled red and white, borehole name with number like BH-1, BH-2 etc.. and Below that you will see elevation, written as "EL:3.65", "EL:6.54" etc...
For every borehole, your job is to find 3 thing:
1. Borehole name... "BH-4"
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
      "Name": "BH-2",
      "Number": 2,
      "Elevation": 102.3
    },
    {
      "Name": "BH-3",
      "Number": 3,
      "Elevation": 99.8
    }
  ]
}

Please return Json only
"""
