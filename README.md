# Car ML
Developed a supervised machine learning model capable of predicting car price.

## SUPERVISED LEARNING: REGRESSION ANALYSIS
-----

### Imports and Data Loading

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn
import sklearn

from sklearn.preprocessing import OneHotEncoder
```

```python
df = pd.read_excel('data/Car.xlsx')
df.head()
```

```
  Location          Maker           Model  Year  Colour  Amount (Million ₦)  \
0    Abuja  Mercedes-Benz         GLA 250  2015   Brown               14.50   
1    Abuja        Hyundai          Accent  2013     Red                1.55   
2    Lagos          Lexus  GX 460 Premium  2011   White               14.00   
3    Lagos          Lexus          ES 350  2011    Gray                4.95   
4   Ibadan         Toyota       Verso 1.6  2009  Silver                1.69   

            Type  Distance_Km  
0   Foreign Used      50000.0  
1  Nigerian Used          NaN  
2   Foreign Used      85000.0  
3   Foreign Used          NaN  
4  Nigerian Used     118906.0
```

-----

### Step #1 Data Cleaning

```python
df.info
```

```
<bound method DataFrame.info of      Location          Maker           Model  Year  Colour  Amount (Million ₦)   
0       Abuja  Mercedes-Benz         GLA 250  2015   Brown               14.50   
1       Abuja        Hyundai          Accent  2013     Red                1.55   
2       Lagos          Lexus  GX 460 Premium  2011   White               14.00   
3       Lagos          Lexus          ES 350  2011    Gray                4.95   
4      Ibadan         Toyota       Verso 1.6  2009  Silver                1.69   
...       ...            ...             ...   ...     ...                 ...   
4482    Lagos          Lexus          RX 330  2006    Blue                4.60   
4483    Lagos          Lexus          ES 350  2007    Blue                4.50   
4484    Abuja  Mercedes-Benz            E350  2014   Green               10.45   
4485    Lagos          Lexus   GX 460 Luxury  2020   Black               31.00   
4486    Lagos          Lexus     RX 450H AWD  2015  Silver               14.00   

            Type  Distance_Km  
0   Foreign Used      50000.0  
1  Nigerian Used          NaN  
2   Foreign Used      85000.0  
3   Foreign Used          NaN  
4  Nigerian Used     118906.0  
...                  ...          ...  
4482                4.60   Foreign Used      90282.0  
4483                4.50   Foreign Used      85000.0  
4484               10.45   Foreign Used      65214.0  
4485               31.00   Foreign Used      45000.0  
4486               14.00   Foreign Used      55000.0  

[4487 rows x 8 columns]>
```

```python
df.isnull().sum()
```

```
Location                 0
Maker                    0
Model                    0
Year                     0
Colour                   0
Amount (Million ₦)       0
Type                     0
Distance_Km           1555
dtype: int64
```

```python
df.describe()
```

```
              Year  Amount (Million ₦)   Distance_Km
count  4487.000000         4487.000000  2.932000e+03
mean   2011.095610           11.309795  1.010383e+05
std       4.823362           20.585915  1.150914e+05
min    1982.000000            0.420000  1.000000e+00
25%    2008.000000            3.600000  5.237850e+04
50%    2011.000000            5.700000  7.900000e+04
75%    2014.000000           12.000000  1.099392e+05
max    2022.000000          454.000000  1.785448e+06
```

```python
# fill up missing values in Distance_Km will the mean
mean_value = df["Distance_Km"].mean()
print(mean_value)

df["Distance_Km"].fillna(mean_value, inplace=True)
```

```
101038.32128240108
```

```python
df.isnull().sum()
```

```
Location              0
Maker                 0
Model                 0
Year                  0
Colour                0
Amount (Million ₦)    0
Type                  0
Distance_Km           0
dtype: int64
```

```python
df.head(10)
```

```
  Location          Maker           Model  Year  Colour  Amount (Million ₦)  \
0    Abuja  Mercedes-Benz         GLA 250  2015   Brown             14.5000   
1    Abuja        Hyundai          Accent  2013     Red              1.5500   
2    Lagos          Lexus  GX 460 Premium  2011   White             14.0000   
3    Lagos          Lexus          ES 350  2011    Gray              4.9500   
4   Ibadan         Toyota       Verso 1.6  2009  Silver              1.6900   
5    Lagos          Lexus          ES 350  2011    Gray              5.8000   
6    Lagos         Toyota  Corolla 1.8 LE  2008   Black              1.9125   
7    Abuja  Mercedes-Benz            E350  2010   Black              4.2000   
8    Lagos  Mercedes-Benz        GL-Class  2014   Black             14.6250   
9    Abuja          Lexus      RX 350 AWD  2012  Silver              9.0000   

            Type    Distance_Km  
0   Foreign Used   50000.000000  
1  Nigerian Used  101038.321282  
2   Foreign Used   85000.000000  
3   Foreign Used  101038.321282  
4  Nigerian Used  118906.000000  
5   Foreign Used  101038.321282  
6  Nigerian Used  115759.000000  
7  Nigerian Used   65000.000000  
8  Nigerian Used  152669.000000  
9  Nigerian Used   80000.000000
```

-----

### Step #2 Feature Engineering

```python
cat_features = ["Model","Year","Type","Colour","Location","Maker"]

for feature in cat_features:
    print(feature,":",df[feature].unique())
    print("##################################################")
```

```
Model:['GLA 250' 'Accent' 'GX 460 Premium' 'ES 350' 'Verso 1.6' 'Corolla 1.8 LE'
 'E350' 'GL-Class' 'RX 350 AWD' 'Land Cruiser 3.5 V6' 'Matrix'
 'Land Cruiser' 'C350' 'Corolla' 'IS 250 4WD' 'Venza V6' 'CX-7' 'RX 350'
 'Highlander Limited 4x4' 'RX' 'RX 350 F Sport AWD' 'Camry'
 'Land Cruiser 5.7 V8 VX-S' 'GLK-Class' 'Avalon' 'GS 300' 'Accord'
 '4-Runner' 'Civic' 'ES 330 Sedan' 'Corolla LE (1.8L 4cyl 2A)' 'Santa Fe'
 'Highlander' 'Elantra' '4-Runner Limited V6' 'Venza Limited FWD V6'
 'M Class ML 350 4Matic' 'M Class' 'Hyundai Kona' 'C300'
 'Camry XLE V6 FWD' 'Range Rover Velar' 'IS 250' 'Highlander Limited'
 'RAV4 Limited FWD' 'Cayenne' 'RX 330' 'RDX' 'Corolla XSE (1.8L 4cyl 2A)'
 'Micra' 'Vibe 2.4L' 'Tacoma TRD Sport' 'Focus 1.8 TDDi Viva' 'GX 460'
 'RAV4 Limited V6 4x4' 'Commander Limited 4x4' 'Tundra' 'Sonata' 'RAV4'
 'F-150' 'GX' 'GS 300 Automatic' 'Sienna' 'Altima' 'Hyundai Ioniq'
 'Tacoma' 'CLA-Class' 'Venza V6 AWD' 'MDX SUV 4dr AWD (3.7 6cyl 5A)'
 'Optima' 'Highlander V6 AWD' 'C250' 206 'GLK-Class 350 4MATIC'
 'ES 350 FWD' 'LX 450d' 'F-150 SuperCab 4x4' 'Avensis' 'Equus' 'ES'
 'Camry XSE (2.5L 4cyl 8A)' 'GLE-Class' 'CR-V' 'C43' 'Sienna LE 4WD'
 'RX 300 4WD' 'Accord 2.4 EX' 'Highlander Sport' 'Venza AWD'
 'Pilot EX-L 4x4 (3.5L 6cyl 5A)' 'Camry SE FWD (2.5L 4cyl 8AM)'
 '6 I Sport' 'Range Rover' 'RX 350 4WD' 'GX 460 Base' 'E550'
 'Pilot EX 4x4 (3.5L 6cyl 5A)' '528i' 'Venza LE AWD V6' 'CR-V LX AWD'
 '5 Series' 'Morning' '328i' 'Edge SE 4dr FWD (3.5L 6cyl 6A)' 'GLA-Class'
 'A-Class a 220 4MATIC' 'RX 350L Luxury FWD' 'Highlander Limited V6 4x4'
 'Highlander Limited V6' 'Accord Sport 2.0T' 'Hilux'
 'GX 470 Sport Utility' 508 'RX 330 AWD' '4-Runner Limited V6 4x4'
 'LX 570 (5 Seats) AWD' 'Land Cruiser Prado 2.8' 'E55' 'Almera'
 'Corolla S' 'Venza' 'Outlander XLS' 'Accord Sedan LX 3 V6 Automatic'
 'Land Cruiser Prado 4 I' 'GL-Class GL 450' 307 'Sportage 2 2WD LX' 'Vibe'
 'RAM' 'Titan Crew Cab SE 4x4' 'LX' 'X Class'
 'Range Rover Evoque Pure Plus AWD' 'Camry XSE FWD (2.5L 4cyl 8AM)'
 'Highlander Sport 4x4' 'Element LX Automatic' 'Genesis 5 RWD'
 'RAV4 Limited V6' 'Odyssey' 'E300' 'LX 570 AWD' 'ES 330' 'Sequoia'
 'Accord EX-L 2.0T' 'MDX' 'Aveo' 'RAV4 2.5 Sport 4x4' 'Explorer' 'iM Base'
 'H300' 'Elantra Limited' 'SLK-Class' 'Highlander 4x4' 'Avalon Limited'
 'X3' '307 2 CC' 'Tacoma TRD Off Road' 'ES 300'
 'Civic 1.4i Sport Automatic' 'Escape' 'A4' '128i' 'Edge' 'C320'
 'G-Class G 65 AMG 4MATIC' 'RX 330 AWD Off' 'RX 330 4WD'
 'Corolla L 4-Speed Automatic' 'Land Cruiser Prado 2.7' 'Wrangler' '535i'
 'Camaro' 'Quest 3.5' 'Challenger' 'IS' 'GS 350' 'LS'
 'Land Cruiser 5.7 V8 VXR' 'RAV4 XLE FWD' 'Venza XLE AWD V6' 'GX 470'
 'RAV4 2 4x4' 'Dyna' 'Corolla Verso 1.8' 'Corolla LE Eco (1.8L 4cyl 2A)'
 'Avensis 2 D Verso' 'Tacoma 4x4 Double Cab'
 '4-Runner Sport Edition 4x4 V6' 'G-Class Base G 550 AWD'
 'S-Class S 500 4MATIC L (V221)' 'Camry XLE FWD'
 'Sienna LE AWD (3.3L V6 5A)' 'G-Class Base G 500 AWD' 'GX 460 Luxury'
 'Land Cruiser Prado' 'CR-V 200i i-VTEC 4x4' 'A6 Avant 2.5 D'
 'Tacoma Limited' '4-Runner Limited 4WD' 'C-Hr Limited FWD' 'NP300'
 'LX 570' 'RAV4 Adventure AWD' 'G35' 'Camry 2.4 LE' 'GL-Class GL 550'
 'RAV4 Sport V6 4x4' 'Solara 3.3 Coupe' 'Clubman Cooper' 'Yaris'
 'RAV4 3.5 Limited 4x4' 'Passat 2 Sedan' 'GLK-Class 350'
 'Highlander XLE AWD' 'RAV4 Limited' 'Corolla Verso 1.8 Luna' 'SX'
 'Land Cruiser Prado 4' 'S-Class S 500 (W222)' 'Fusion SE AWD'
 'RX 350L Luxury AWD' 'X5 xDrive40i AWD' 'E430' 'GS' 'Civic HF Sedan EX'
 'Honda HR-V' 'GL Class' '300 300C RWD' 'Corolla LE'
 'Range Rover Evoque Matt' 'Camry LE FWD (2.5L 4cyl 8AM)' 'GX 470 Off'
 'G-Class' 'Hilux 2 VVT-i' 'Xterra SE' 'Trailrunner'
 'Land Cruiser 4.5 V8 GXR' 'RX 300 2WD' 'C240' 'C400'
 'RAV4 XLE Premium AWD' 'Hilux SR 4x4' '230E' 'CL' 'Almera Tino' 'Passat'
 'S-Class Off' 'Range Rover Evoque' 'C-Class C 300 CDI 4MATIC (W204)'
 '4-Runner Limited 4x4 V6' 'Avalon XLS' 'Land Cruiser HDJ 100'
 'Sharan 1.8 T' 'C70 Automatic' 'RX 450h' 'Veloster Automatic' 'GLS-Class'
 'Ridgeline RTX' 'ZDX Base AWD' 'Pilot' 'Sienna CE AWD'
 'Land Cruiser Prado 2.7 VVT-i' 'ES 250' 'Corolla Liftback'
 'Tacoma Access Cab' 'Cooper Cabriolet' 'Sportage 2.0L Automatic'
 'Edge SE 4dr AWD (3.5L 6cyl 6A) Off' 'Venza Limited AWD V6' 'CLS'
 'Sentra 2 SL' 'B-Class' 'CR-V Touring AWD' 'X5 3.0i' 'Range Rover Vogue'
 'Almera 1.6 Lux Automatic' 'Venza XLE AWD' 'Rolls-Royce Ghost Matt'
 'S-Class S 550' 'Sienna XLE 7 Passenger' 'TL Automatic' 'Corolla SE'
 'Altima 2.5' 'Sentra' 'X-Trail' 'Accord Sedan LX'
 'Land Cruiser 5.7 V8 VX-S Matt' 'RAV4 2 VVT-i' '4-Runner Limited'
 'G-Class G 63 AMG' 3008 'Range Rover Sport HSE 4x4 (5.0L 8cyl 6A)'
 'RX 300' 'Hilux WORKMATE 4x4' 'Sharan' 'Genesis 3.8' 'TL SH-AWD'
 'Rover Discovery' 'MDX 3.5L 4x4' 'MDX Tech & A-Spec Pkgs SH-AWD'
 'RX 450hL Luxury AWD' 'Cruze' 'Sienna CE' 'Golf' '7 Series'
 'Highlander Limited V6 FWD' 'LX 570 Three-Row' 'Aveo 1LT' 'SpaceWagon'
 406 'GLC-Class' 'A4 3 TDI Automatic' 'JAC S2' 'Yaris 1.3 VVT-i'
 'Corolla 1.4' 'Tiida' 'Highlander Limited 3.5l 4WD' 'Genesis 4.6'
 'Corolla 1.8 VVTL-i TS' 'L200 Double Cab 2.5 180hp 4WD' 'Highlander V6'
 'RX 350 FWD' 'Avalon 3' 'Carens' 'RAV4 2.2 D-4D GX' 'F-150 SuperCrew'
 'Odyssey 2.4 4WD' 'RAV4 Sport' 'Elantra 1.6 GL' 'Ridgeline'
 'Golf 2 GL 3-Door' 'Camry XSE V6 (3.5L V6 8A)' 'Sienna XLE'
 'Accord 2 Comfort Automatic' 'Forester 2.5X Limited' '4-Runner Off'
 'HS 250h' 'A6 2' 'JX 35' 'CR-V EX-L 4WD Automatic' 'Duster' 'Jetta'
 'Highlander SE 3.5L 4WD' 'Sienna CE FWD (3.3L V6 5A)'
 'Rolls-Royce Cullinan Base' 'Accord Sedan EX Automatic'
 'CR-V 2.4 EX Automatic' 'Hilux 2.4 Diesel' 'RX 400h'
 'M Class ML350 AWD 4MATIC' 'Sienna LE 7 Passenger' 'IS 350'
 'Hilux 2.7 VVT-i 4X4 SRX' 'RC 350 AWD' 'C63' 'M Class ML 320'
 'Freestar Wagon SE' 'M Class ML 350' 'IS 250 AWD'
 'Solara 3.3 Convertible' 'Camry 2.4 SE Automatic' 'Land Cruiser 3.3 DT'
 'RX 200t 2WD' 'Tacoma X Runner' 'CR-V EX Automatic' 'Elantra GLS'
 'Acadia SLE AWD' 'Rio' '3 Series' 'Fortuner' 'RAV4 3.5 Sport 4x4'
 'Mustang' 'IS 350 C' 'RAV4 3.5 Limited' 'Tacoma Access Cab V6 Automatic'
 'RAV4 LE AWD' 'M Class ML 350 4x2' 'Tucson' 'GS 250'
 'Corolla SE (1.8L 4cyl 2A)' 'Land Cruiser Base 4x4' 3
 'Land Cruiser 4 V6 GXR' 'Zafira' 'Corolla 1.4 D-4d' 'Civic 1.8i VTEC'
 'Corolla CE' '4-Runner SR5 4WD' 'C-Hr' 'Beetle' 626
 'G-Class G 63 AMG 4MATIC' 'IS 250 AWD Automatic' 'Accord Sedan LX SE'
 'Corolla 1.8 Exclusive Automatic' 'X5 3.0i Sport' 'Sienna XLE AWD'
 'Traverse 1LT' 'Odyssey 2.4 2WD' 'MDX Base FWD' 'SL-Class'
 'Tacoma Double Cab V6 4WD' 'Accord Automatic' 'Ranger XL' 'S40 a'
 'Civic 1.8 DX' 'Qashqai' 'Cerato 1.6 LX Automatic'
 'GX 470 Sport Utility Off' 'Bentley Continental' 'Sienna LE AWD'
 'Sportage' 'Land Cruiser 5.7 V8 EXR' 'Highlander V6 4x4'
 'Santa Fe Limited' 'Pathfinder' 'RX 350 XE 4x4' 'Honda Accord' 'Picanto'
 '6 Wagon' 'RX 350 4x4' 'A7' 'Accord Crosstour EX'
 'Highlander Limited 3.5L 2WD' 'Tucson Limited AWD' 'Countryman S All4'
 'MKS EcoBoost' 'Land Cruiser Prado 3.4' 'F-150 Lariat 4x4' 'M5'
 'Highlander 3.5L V6 4WD' 'Venza LE FWD V6' 'Pilot Touring 4x4 (3.5L 6cyl 5A)'
 'Outlander SE 4dr 4WD (2.4L 4cyl CVT)' 'Avalon Touring' 'Hilux 3.0 D-4D'
 'X5 M' 'Matrix 2.4 S' 'F-150 XL SuperCrew 4x4' 'Highlander 3.5l AWD'
 'RX 350 F Sport' 'M-Class' 'X3 xDrive28i AWD' 'GL-Class GL 450 AWD'
 'Odyssey Touring (3.5L 6cyl 5A)' 'RAV4 LE FWD' 'XC90'
 'Tacoma Double Cab' '4-Runner SR5 V6 4x4' 'Land Cruiser Prado 3.0 D-4D'
 'Corolla S 4-Speed Automatic' 'Highlander 4x2' 'RAV4 XLE AWD'
 'Accent Blue 4dr Sedan' 'Pilot EX' 'Range Rover Sport'
 'CR-V SE 4WD Automatic' 'E-Class E 450' '3.5L V6 AWD' 'A4 Avant 2.0 T'
 'Corolla 1.6 VVT-i Luna' 'A-Class' 'Pilot EX-L 4x4' 'Tucson SE AWD'
 'Focus' 'Range Rover Evoque SE Dynamic 4x4 5-Door' 'X5'
 'Range Rover Evoque HSE 4x4 5-Door' 'Corolla 1.6'
 'Range Rover Sport HSE' 'Tacoma TRD Off Road 4x4' 'E-Class'
 'Corolla 1.6 VVT-i Sol' 'Corolla 1.6 VVT-i' 'Santa Fe Sport FWD'
 'RX 450H AWD' 'Golf GTI' 'Challenger R/T CLASSIC' '206 GTi 180'
 'Cooper S' 'CR-V 2 Automatic' 'Elantra 1.6 Automatic'
 'Hyundai Sonata SEL Plus' 'GLK-Class 350 SUV'
 'Accord Coupe 3.5 EX-L V6' 'MDX Base SH-AWD' 'Lamborghini Urus'
 'CX-9 Sport' 'Micra Visia 1.2' 'Mustang V6 Premium' 'RAV4 V6'
 'Land Cruiser Prado 3.4 5dr' 'F-150 SuperCab' 'Primera' 'Rogue' 'A8'
 'Avalon XSE' 'Pilot EX 4dr SUV (3.5L 6cyl 5A)' 'Sienna LE'
 'RAV4 2.5 AWD' 'Avalon XL' 'Escape 2.5L I4 FWD' 'G-Class G 550'
 'Corolla 1.8 Terra' 'Tacoma 4x4 Access Cab' 'RAV4 XLE FWD (2.5L 4cyl 8A)'
 'Edge Limited 4dr AWD (3.5L 6cyl 6A)' 'S-Class' 'GLK-Class 280 4MATIC'
 'Land Cruiser 3.0 D' 'Venza XLE V6' 'Tacoma Base 4x4'
 'GLC-Class GLC 300 4MATIC' 'RAV4 XLE FWD (2.5L 4cyl 6A)'
 'G-Class G 500 L 4MATIC' 'X1' 'Corolla 1.8 SE' 'LX 470' 'C200'
 'Corolla 1.8 XRS' 'Highlander 3.5l' 'RAV4 2.0 4WD' 'RAV4 LE'
 'RAV4 Limited AWD' 'RAV4 2.0 VVT-i' 'RAV4 2.0 D-4D'
 'RAV4 Limited V6 FWD' 'Land Cruiser 5.7 V8' 'F-150 Lariat'
 'Camry LE FWD' 'Tundra Double Cab SR5 4x4' 'Sienna XLE AWD 7 Passenger'
 'RAV4 2.4' 'F-150 SuperCrew Cab' 'Tacoma TRD Sport Double Cab 4x4'
 'CR-V EX 4WD Automatic' 'Avalon XLE Touring Limited' 'RX 300 AWD Off'
 'Charger R/T' 'CLK' 'Highlander SE 4x4 V6 (3.5L 6cyl 8A)'
 '3 2.5 S Grand Touring Sedan' 'Ridgeline RTL'
 'Range Rover Velar P380 HSE R-Dynamic 4x4' 'F-150 SuperCrew 4x4'
 'Hyundai Palisade' 'Civic LX Hatchback' 'Camry SE (2.5L 4cyl 8A)' 'Sunny'
 'Beetle 2.5' 'ES 350 Luxury FWD' 'Sienna XLE Limited'
 'Accord 3.5 EX Automatic' 'Forte SX Hatchback' 'Express Cargo Van 1500'
 'Explorer Sport Track Automatic' 'Hilux SR5+ 4x4'
 'FJ Cruiser 4x4 Automatic' 'Sorento EX 4dr SUV (3.3L 6cyl 6A)' 1117
 'Santa Fe Sport 2.0T Off' 'Outlander 2.4' 'Avanza'
 'GLS-Class GLS63 AMG Base' 'M Class ML350 4x2' 'X5 3.0si Activity'
 'Tacoma Double Cab V6' 'Escalade' 'Corolla 1.8 CE'
 'Sienna XLE 7-Passenger AWD' '4-Runner Limited 2WD' 'C280' 'E-350'
 'GL Class GL 550' 'Peugeot 5008 2 HDi 180 GT' 'Nissan Armada' 'M6' 'TL'
 'C-HR' 'Accord Coupe EX-L' 'Range Rover Evoque HSE Dynamic 4x4 5-Door'
 'RAV4 Automatic' 'CX-9 Grand Touring' 'Charger' 'Camry Off'
 '407 2 HDi ST Comfort' 'Versa 1.8 SL Hatchback' 'Land Cruiser Matt'
 'Yaris Sedan' 'Camry 2.4 LE Automatic' 'GX 460 Off' 'M Class ML 350 4x4'
 '407 2 HDi' 'X5 4.8i Sports Activity' 'Elantra 2' 'F-150 Super Cab 4x4' 'Navigator'
 'CX-9' 'CR-V LX 4WD Automatic' 'Outback']
##################################################
Year:[2015 2013 2011 2009 2008 2010 2014 2012 2022 2006 2021 2017 2007 2002
 2016 2019 2020 2004 2018 2005 2003 2000 1999 2001 1989 1998 1982 1994
 1993 1997]
##################################################
Type:['Foreign Used' 'Nigerian Used' 'Brand New']
##################################################
Colour:['Brown' 'Red' 'White' 'Gray' 'Silver' 'Black' 'Blue' 'Gold' 'Green'
 'Beige' 'Purple' 'Orange' 'Burgandy' 'Ivory' 'Pink' 'Pearl' 'Yellow'
 'Luury' 'Teal']
##################################################
Location:['Abuja' 'Lagos' 'Ibadan']
##################################################
Maker:['Mercedes-Benz' 'Hyundai' 'Lexus' 'Toyota' 'Mazda' 'Honda' 'Land Rover'
 'Porsche' 'Acura' 'Nissan' 'Pontiac' 'Ford' 'Jeep' 'Kia' 'Peugeot' 'BMW'
 'Mitsubishi' 'Dodge' 'Chevrolet' 'Scion' 'Audi' 'Infiniti' 'Mini'
 'Volkswagen' 'Suzuki' 'Chrysler' 'Volvo' 'Rolls-Royce' 'JAC' 'Subaru'
 'Renault' 'GMC' 'Rover' 'IVM' 'Bentley' 'Opel' 'Lincoln' 'Hummer'
 'Saturn' 'Cadillac' 'Lamborghini' 'Buick' 'Smart' 'Jaguar' 'Ferrari'
 'Tata' 'Skoda']
##################################################
```

```python
# Drop the Model feature
df.drop("Model", axis=1, inplace=True)
df.head()
```

```
  Location          Maker  Year  Colour  Amount (Million ₦)           Type  \
0    Abuja  Mercedes-Benz  2015   Brown               14.50   Foreign Used   
1    Abuja        Hyundai  2013     Red                1.55  Nigerian Used   
2    Lagos          Lexus  2011   White               14.00   Foreign Used   
3    Lagos          Lexus  2011    Gray                4.95   Foreign Used   
4   Ibadan         Toyota  2009  Silver                1.69  Nigerian Used   

     Distance_Km  
0   50000.000000  
1  101038.321282  
2   85000.000000  
3  101038.321282  
4  118906.000000
```

-----

### Step #3 Encoding

```python
# Label Encoding
cat_features = ["Location","Maker","Year","Colour","Type"]

for feature in cat_features:
    df[feature + "_cat"] = df[feature].astype("category").cat.codes

df.head()
```

```
  Location          Maker  Year  Colour  Amount (Million ₦)           Type  \
0    Abuja  Mercedes-Benz  2015   Brown               14.50   Foreign Used   
1    Abuja        Hyundai  2013     Red                1.55  Nigerian Used   
2    Lagos          Lexus  2011   White               14.00   Foreign Used   
3    Lagos          Lexus  2011    Gray                4.95   Foreign Used   
4   Ibadan         Toyota  2009  Silver                1.69  Nigerian Used   

     Distance_Km  Location_cat  Maker_cat  Year_cat  Colour_cat  Type_cat  
0   50000.000000             0         26        22           3         1  
1  101038.321282             0         14        20          14         2  
2   85000.000000             2         23        18          17         1  
3  101038.321282             2         23        18           6         1  
4  118906.000000             1         44        16          15         2
```

```python
# Drop the reductant features since Label encoding have been done
df.drop(["Location","Maker","Year","Colour", "Type"], axis=1, inplace=True)
df.head()
```

```
   Amount (Million ₦)    Distance_Km  Location_cat  Maker_cat  Year_cat  \
0             14.5000   50000.000000             0         26        22   
1              1.5500  101038.321282             0         14        20   
2             14.0000   85000.000000             2         23        18   
3              4.9500  101038.321282             2         23        18   
4              1.6900  118906.000000             1         44        16   

   Colour_cat  Type_cat  
0           3         1  
1          14         2  
2          17         1  
3           6         1  
4          15         2
```

-----

### Step #4 Perform data segmentation

```python
X = df.drop("Amount (Million ₦)", axis=1)
y = df["Amount (Million ₦)"]
```

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

-----

## Step #5 Model Building

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

regressor = Pipeline([
    ("scaler", StandardScaler())
])
regressor
```

```
<pre>Pipeline(steps=[('scaler', StandardScaler())])</pre>
```

```python
x_train = regressor.fit_transform(x_train)
x_test = regressor.transform(x_test)
```

-----

## Step #6 Train your model

```python
from sklearn.linear_model import LinearRegression
```

```python
reg = LinearRegression()
```

```python
reg.fit(x_train, y_train)
```

```
<pre>LinearRegression()</pre>
```

```python
from sklearn.metrics import r2_score

y_pred = reg.predict(x_test)
print("R2 Score", r2_score(y_test, y_pred))
```

```
R2 Score 0.4077598506168541
```

```python
reg.predict(x_test)
```

```
array([ 3.32709939, 11.23126756, 12.33800262, ..., 15.65997233,
       10.99263641,  5.80808544])
```

-----

## Step #7 Evaluate your model

```python
from sklearn.metrics import mean_absolute_error

# we are using mean_absolute_error because
# this is a regression model

y_pred = reg.predict(x_test)

print("MAE",mean_absolute_error(y_test,y_pred))
```

```
MAE 7.617421595973497
```

-----

### Ridge Regression

```python
from sklearn.linear_model import Ridge

ridge = Ridge()
```

```python
ridge.fit(x_train, y_train)
```

```
<pre>Ridge()</pre>
```

```python
y_pred = ridge.predict(x_test)
print("R2 Score", r2_score(y_test, y_pred))
```

```
R2 Score 0.4077674697395027
```

```python
y_pred = ridge.predict(x_test)
print("MAE",mean_absolute_error(y_test,y_pred))
```

```
MAE 7.616781442475965
```

-----

### Lasso Regression

```python
from sklearn.linear_model import Lasso

lasso = Lasso()
```

```python
lasso.fit(x_train, y_train)
```

```
<pre>Lasso()</pre>
```

```python
y_pred = lasso.predict(x_test)
print("R2 Score", r2_score(y_test, y_pred))
```

```
R2 Score 0.4005881473957827
```

```python
y_pred = lasso.predict(x_test)
print("MAE",mean_absolute_error(y_test,y_pred))
```

```
MAE 7.426296313936495
```

-----

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rfR = RandomForestRegressor()
rfR.fit(x_train, y_train)
y_pred = rfR.predict(x_test)
print("MAE",mean_absolute_error(y_test,y_pred))
```

```
MAE 4.469956528080825
```

-----

### Model Saving

```python
import joblib

joblib.dump(rfR,"RandomForest_model.pkl")

print("model saved!")
```

```
model saved!
```

## 👨🏽‍💻 Author

Japhet Ujile
📧 [assistant.rawlings@gmail.com](mailto:assistant.rawlings@gmail.com)
🌐 [GitHub](https://github.com/assistantrawlings-lgtm) | [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/japhet-ujile-838442148%3Futm_source%3Dshare%26utm_campaign%3Dshare_via%26utm_content%3Dprofile%26utm_medium%3Dandroid_app%5D)
