# 2020-1-OSS Team 10
- [Static Page](#static_page)
- [Team](#about_team)
- [Our Project](#about_project) 
  - [Type of Graph](#graph)
  - [Color Change](#color_change)
  - [Notes:](#notes)
- [Team Progress](#progress)

## <div id="static_page">Static Page</div>
https://20-1-skku-oss.github.io/2020-1-OSS-10/

## <div id="about_team">Team</div>
### Team leader
- 서주원<br>
>Github : [Bearhunter49](https://github.com/BreadHunter49) <br><br>

### Team members
- 주원규 <br>
>Github : [jwk6553](https://github.com/jwk6553) <br><br>

- 장병우<br>
>Github : [onion-Ring](https://github.com/onion-Ring) <br><br>

- 이지영<br>
>Github : [ljy719](https://github.com/ljy719) <br><br>

- 김광원<br>
>Github : [one0955](https://github.com/one0955)


<br>
<br>
<br>

## <div id="about_project">Visuaudio</div>
### A fun GUI application to visualize audio spectrum
<b>https://github.com/irahorecka/visuaudio</b>

```visuaudio.py``` uses the ```pyqtgraph``` and ```pyaudio``` libraries to view the audio spectrum of input sound.
<hr>

<b>Running the application:</b>
1) Clone repository
2) ```$ pip install -r requirements.txt```<br>
3) ```$ python visuaudio.py```
<br>

## <div id="graph">Type of Graph</div>

### Bar Graph
<p align="center">
<img src=https://i.imgur.com/pIpCaUQ.png alt="Audio Spectrum GUI"
    width=800>
</p>

### Scatter Graph(added)
<p align="center">
<img src=https://github.com/20-1-SKKU-OSS/2020-1-OSS-10/blob/master/img/scatter.png?raw=true"
    width=800>
</p>
                                                                                             
### Curve Graph(added)
<p align="center">
<img src=https://github.com/20-1-SKKU-OSS/2020-1-OSS-10/blob/master/img/curve.png?raw=true"
    width=800>
</p>                                                                                           
                                                                                           
### Line Graph(added)
<p align="center">
<img src=https://github.com/20-1-SKKU-OSS/2020-1-OSS-10/blob/master/img/LineGraph.PNG?raw=true"
    width=800>
</p>
<b>       
<br>
                                                                                               
## <div id="color_change">Color Change</div>

#### You can change color of graph in running.
- ↑,→,←: R,G,B += 10
- F1 ~ F9: White ~ Pink
<br>

## <div id="notes">Notes:</div>
<ul>
<li>Ensure you have a working input sound source.(ex. Mic)</li>
<li>Run the application on your native terminal (i.e. not iTerm2, etc.)</li>
<li>You should run on Python version > 3 </li>
</ul>                                                                                           
<b>MacOS</b>
<ul>
    <li>You will have to grant Terminal permission to use the input sound source.</li>
</ul>              
<b>Windows</b>
<ul>                                                                                           
    <li>Go link: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio</li>
    <li>You should download <b>'PyAudio-0.2.11-cp{your python version}-cp{your python version}m-win_amd64.whl'</b></li>
    <li> > $ pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl</li>
</ul>
<b>Ubuntu Linux</b>

<br><br>
                                                                                           
## <div id="progress">Team Progress</div>
- Added 3 more types of visualization
  - Scatter
  - Curve
  - Line
- Added color selection
- Added symbol selection
- Added exit function
