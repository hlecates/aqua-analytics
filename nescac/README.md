# NESCAC Swimming Data Project

## Why Manual Data Entry is Needed

Although the parsing pipeline has decent performance—successfully parsing approximately 24 entries for every finals event—there are still significant issues due to the irregularity of recorded data across formats and the nature of swimming results themselves. Because the dataset consists of only one event per year, the sample size is small and it is not feasible to simply drop problematic events or tolerate mutated data. 

To address this, the `manual_update` pipeline was developed. This workflow allows the user to:
- Convert parsed CSV files into human-readable text files for easy manual review and correction.
- Edit and clean up event and swimmer data directly, ensuring accuracy and consistency.
- Convert the corrected text files back into CSV format, overwriting the originals and maintaining a clean dataset.

This approach ensures that all events are preserved and the data quality is as high as possible, even when automated parsing fails or produces errors due to inconsistent source formatting.

Some name were cutoff in the prelims such as kearns --> kear in the prelims in 2002, to many to simply hardcode, and

---

https://nescac.com/sports/2020/7/14/championships-pastchamps-msd.aspx


should be cleaner data, need to try multiple parsing techniques for each results

some have results in tables on html
some have two col pdfs
and others have 


Had to drop 2002, 2003, 2004, 2008, since no seed time were included, hence no featured would be able to be created for the targets

Use a "case study"

I am mid distance swimmer who swims fly and freestyle. The 100 fly and 200 free are on the saem day, hence need to choose one, and want ot make the biggest impact via scoring points for my team. My PR in 200 free is XX, season best was XX, same for 100 fly. Via the models predictions, both the simple and advancded model, would have predicted me missing the C final of the 200 free, and failing to make finals and hence score points, BUT make the B final of the 100 butterfly, allowinf for pints to be scored

Using the 2024 predictions and related times, the advanced model would predict me to make the b final with my PB, and C final with season best. Since nescac is a championnship meet, it is most likely yhe season best time will be imporved upin hence, use the PB. Then for the 100 fly with PB it would predicit me to be make the A final, and season best would make the C final. This has a higher potential pint output, hence sswim the 100 fly at nescacs.

2024
200 FREE:
    PB: 1:44.13
    Season PB: 1:45.61

100 FLY
    PB: 49.81
    Season PB: 51.01

2025
200 FREE:
    PB: 1:43.06
    Season PB: 1:46.16

100 FLY
    PB: 49.46 
    Season PB: 51.15

TODO
convert the parsing and pipeling and modeling scripts to create and populate a db instance. --> this is really only necessary when more backend features need querying

Allow for some way for app users to add more meet results


