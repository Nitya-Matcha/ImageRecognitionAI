import os
import shutil

os.makedirs(images, exist = True) 
os.cddir(images) 
'''
I'm pretty sure the above code that says os.cddir images is highkey wrong, 
because it's supposed to contain brackets but I'm unsure whether I'm supposed to be calling the main directory or the subdirectories
'''
#I'm making this a comment, I dont think this is how you make subdirectories os.makedirs(Myocardial, Abnormal, History, Normal)

#Tbh the code below I got from an online source because like creating subdirectories for everything and their mom was kind of a pain

for name in names:
  classdirectory = os.path.join(images, name)
  os.makedirs(classdirectory, exist = True)
  for imagepaths in imagepathsforallclasses(name, []):
    shutil.copy(imagepaths, classdirectory)
else:
  print("Sorry, couldn't copy image path :(")

#I'm getting an error here that says unexpected Indent, but tbh I think all my indents r fine
if __name__ == "__main__":
  images = "images"
  names = ["Myocardial", "Abnormal", "History", "Normal"]

  imagesinclass = {
    "Myocardial": ["/path/to/MI(1).jpg", "/path/to/MI(2).jpg", "/path/to/MI(3).jpg", "/path/to/MI(4).jpg", "/path/to/MI(5).jpg", "/path/to/MI(6).jpg", "/path/to/MI(7).jpg", "/path/to/MI(8).jpg", "/path/to/MI(9).jpg", "/path/to/MI(10).jpg", "/path/to/MI(11).jpg", "/path/to/MI(12).jpg", "/path/to/MI(13).jpg", "/path/to/MI(14).jpg", "/path/to/MI(15).jpg", "/path/to/MI(16).jpg", "/path/to/MI(17).jpg", "/path/to/MI(18).jpg", "/path/to/MI(19).jpg", "/path/to/MI(20).jpg", "/path/to/MI(21).jpg", "/path/to/MI(22).jpg",
    "/path/to/MI(23).jpg", "/path/to/MI(24).jpg", "/path/to/MI(25).jpg", "/path/to/MI(26).jpg", "/path/to/MI(27).jpg", "/path/to/MI(28).jpg", "/path/to/MI(29).jpg", "/path/to/MI(30).jpg", "/path/to/MI(31).jpg", "/path/to/MI(32).jpg", "/path/to/MI(33).jpg", "/path/to/MI(34).jpg", "/path/to/MI(35).jpg", "/path/to/MI(36).jpg", "/path/to/MI(37).jpg", "/path/to/MI(38).jpg", "/path/to/MI(39).jpg", "/path/to/MI(40).jpg", "/path/to/MI(41).jpg", "/path/to/MI(42).jpg", "/path/to/MI(43).jpg", "/path/to/MI(44).jpg",
    "/path/to/MI(45).jpg", "/path/to/MI(46).jpg", "/path/to/MI(47).jpg", "/path/to/MI(48).jpg", "/path/to/MI(49).jpg", "/path/to/MI(50).jpg", "/path/to/MI(51).jpg", "/path/to/MI(52).jpg", "/path/to/MI(53).jpg", "/path/to/MI(54).jpg", "/path/to/MI(55).jpg", "/path/to/MI(56).jpg", "/path/to/MI(57).jpg", "/path/to/MI(58).jpg", "/path/to/MI(59).jpg", "/path/to/MI(60).jpg", "/path/to/MI(61).jpg", "/path/to/MI(62).jpg", "/path/to/MI(63).jpg", "/path/to/MI(64).jpg", "/path/to/MI(65).jpg", "/path/to/MI(66).jpg", 
    "/path/to/MI(67).jpg", "/path/to/MI(68).jpg", "/path/to/MI(69).jpg", "/path/to/MI(70).jpg", "/path/to/MI(71).jpg", "/path/to/MI(72).jpg", "/path/to/MI(73).jpg", "/path/to/MI(74).jpg", "/path/to/MI(75).jpg", "/path/to/MI(76).jpg", "/path/to/MI(77).jpg", "/path/to/MI(78).jpg", "/path/to/MI(79).jpg", "/path/to/MI(80).jpg", "/path/to/MI(81).jpg", "/path/to/MI(82).jpg", "/path/to/MI(83).jpg", "/path/to/MI(84).jpg", "/path/to/MI(85).jpg", "/path/to/MI(86).jpg", "/path/to/MI(87).jpg", "/path/to/MI(88).jpg", 
    "/path/to/MI(89).jpg", "/path/to/MI(90).jpg", "/path/to/MI(91).jpg", "/path/to/MI(92).jpg", "/path/to/MI(93).jpg", "/path/to/MI(94).jpg", "/path/to/MI(95).jpg", "/path/to/MI(96).jpg", "/path/to/MI(97).jpg", "/path/to/MI(98).jpg", "/path/to/MI(99).jpg", "/path/to/MI(100).jpg", "/path/to/MI(101).jpg", "/path/to/MI(102).jpg", "/path/to/MI(103).jpg", "/path/to/MI(104).jpg", "/path/to/MI(105).jpg", "/path/to/MI(106).jpg", "/path/to/MI(107).jpg", "/path/to/MI(108).jpg", "/path/to/MI(109).jpg", "/path/to/MI(110).jpg", 
    "/path/to/MI(111).jpg", "/path/to/MI(112).jpg", "/path/to/MI(113).jpg", "/path/to/MI(114).jpg", "/path/to/MI(115).jpg", "/path/to/MI(116).jpg", "/path/to/MI(117).jpg", "/path/to/MI(118).jpg", "/path/to/MI(119).jpg", "/path/to/MI(120).jpg"],

    "Abnormal": ["/path/to/HB(1).jpg", "/path/to/HB(2).jpg", "/path/to/HB(3).jpg", "/path/to/HB(4).jpg", "/path/to/HB(5).jpg", "/path/to/HB(6).jpg", "/path/to/HB(7).jpg", "/path/to/HB(8).jpg", "/path/to/HB(9).jpg", "/path/to/HB(10).jpg", "/path/to/HB(11).jpg", "/path/to/HB(12).jpg", "/path/to/HB(13).jpg", "/path/to/HB(14).jpg", "/path/to/HB(15).jpg", "/path/to/HB(16).jpg", "/path/to/HB(17).jpg", "/path/to/HB(18).jpg", "/path/to/HB(19).jpg", "/path/to/HB(20).jpg", "/path/to/HB(21).jpg", "/path/to/HB(22).jpg",
    "/path/to/HB(23).jpg", "/path/to/HB(24).jpg", "/path/to/HB(25).jpg", "/path/to/HB(26).jpg", "/path/to/HB(27).jpg", "/path/to/HB(28).jpg", "/path/to/HB(29).jpg", "/path/to/HB(30).jpg", "/path/to/HB(31).jpg", "/path/to/HB(32).jpg", "/path/to/HB(33).jpg", "/path/to/HB(34).jpg", "/path/to/HB(35).jpg", "/path/to/HB(36).jpg", "/path/to/HB(37).jpg", "/path/to/HB(38).jpg", "/path/to/HB(39).jpg", "/path/to/HB(40).jpg", "/path/to/HB(41).jpg", "/path/to/HB(42).jpg", "/path/to/HB(43).jpg", "/path/to/HB(44).jpg",
    "/path/to/HB(45).jpg", "/path/to/HB(46).jpg", "/path/to/HB(47).jpg", "/path/to/HB(48).jpg", "/path/to/HB(49).jpg", "/path/to/HB(50).jpg", "/path/to/HB(51).jpg", "/path/to/HB(52).jpg", "/path/to/HB(53).jpg", "/path/to/HB(54).jpg", "/path/to/HB(55).jpg", "/path/to/HB(56).jpg", "/path/to/HB(57).jpg", "/path/to/HB(58).jpg", "/path/to/HB(59).jpg", "/path/to/HB(60).jpg", "/path/to/HB(61).jpg", "/path/to/HB(62).jpg", "/path/to/HB(63).jpg", "/path/to/HB(64).jpg", "/path/to/HB(65).jpg", "/path/to/HB(66).jpg", 
    "/path/to/HB(67).jpg", "/path/to/HB(68).jpg", "/path/to/HB(69).jpg", "/path/to/HB(70).jpg", "/path/to/HB(71).jpg", "/path/to/HB(72).jpg", "/path/to/HB(73).jpg", "/path/to/HB(74).jpg", "/path/to/HB(75).jpg", "/path/to/HB(76).jpg", "/path/to/HB(77).jpg", "/path/to/HB(78).jpg", "/path/to/HB(79).jpg", "/path/to/HB(80).jpg", "/path/to/HB(81).jpg", "/path/to/HB(82).jpg", "/path/to/HB(83).jpg", "/path/to/HB(84).jpg", "/path/to/HB(85).jpg", "/path/to/HB(86).jpg", "/path/to/HB(87).jpg", "/path/to/HB(88).jpg", 
    "/path/to/HB(89).jpg", "/path/to/HB(90).jpg", "/path/to/HB(91).jpg", "/path/to/HB(92).jpg", "/path/to/HB(93).jpg", "/path/to/HB(94).jpg", "/path/to/HB(95).jpg", "/path/to/HB(96).jpg", "/path/to/HB(97).jpg", "/path/to/HB(98).jpg", "/path/to/HB(99).jpg", "/path/to/HB(100).jpg", "/path/to/HB(101).jpg", "/path/to/HB(102).jpg", "/path/to/HB(103).jpg", "/path/to/HB(104).jpg", "/path/to/HB(105).jpg", "/path/to/HB(106).jpg", "/path/to/HB(107).jpg", "/path/to/HB(108).jpg", "/path/to/HB(109).jpg", "/path/to/HB(110).jpg",
    "/path/to/HB(111).jpg", "/path/to/HB(112).jpg", "/path/to/HB(113).jpg", "/path/to/HB(114).jpg", "/path/to/HB(115).jpg", "/path/to/HB(116).jpg", "/path/to/HB(117).jpg", "/path/to/HB(118).jpg", "/path/to/HB(119).jpg", "/path/to/HB(120).jpg"],

    "History": ["/path/to/PMI(1).jpg", "/path/to/PMI(2).jpg", "/path/to/PMI(3).jpg", "/path/to/PMI(4).jpg", "/path/to/PMI(5).jpg", "/path/to/PMI(6).jpg", "/path/to/PMI(7).jpg", "/path/to/PMI(8).jpg", "/path/to/PMI(9).jpg", "/path/to/PMI(10).jpg", "/path/to/PMI(11).jpg", "/path/to/PMI(12).jpg", "/path/to/PMI(13).jpg", "/path/to/PMI(14).jpg", "/path/to/PMI(15).jpg", "/path/to/PMI(16).jpg", "/path/to/PMI(17).jpg", "/path/to/PMI(18).jpg", "/path/to/PMI(19).jpg", "/path/to/PMI(20).jpg", "/path/to/PMI(21).jpg", "/path/to/PMI(22).jpg",
    "/path/to/PMI(23).jpg", "/path/to/PMI(24).jpg", "/path/to/PMI(25).jpg", "/path/to/PMI(26).jpg", "/path/to/PMI(27).jpg", "/path/to/PMI(28).jpg", "/path/to/PMI(29).jpg", "/path/to/PMI(30).jpg", "/path/to/PMI(31).jpg", "/path/to/PMI(32).jpg", "/path/to/PMI(33).jpg", "/path/to/PMI(34).jpg", "/path/to/PMI(35).jpg", "/path/to/PMI(36).jpg", "/path/to/PMI(37).jpg", "/path/to/PMI(38).jpg", "/path/to/PMI(39).jpg", "/path/to/PMI(40).jpg", "/path/to/PMI(41).jpg", "/path/to/PMI(42).jpg", "/path/to/PMI(43).jpg", "/path/to/PMI(44).jpg",
    "/path/to/PMI(45).jpg", "/path/to/PMI(46).jpg", "/path/to/PMI(47).jpg", "/path/to/PMI(48).jpg", "/path/to/PMI(49).jpg", "/path/to/PMI(50).jpg", "/path/to/PMI(51).jpg", "/path/to/PMI(52).jpg", "/path/to/PMI(53).jpg", "/path/to/PMI(54).jpg", "/path/to/PMI(55).jpg", "/path/to/PMI(56).jpg", "/path/to/PMI(57).jpg", "/path/to/PMI(58).jpg", "/path/to/PMI(59).jpg", "/path/to/PMI(60).jpg", "/path/to/PMI(61).jpg", "/path/to/PMI(62).jpg", "/path/to/PMI(63).jpg", "/path/to/PMI(64).jpg", "/path/to/PMI(65).jpg", "/path/to/PMI(66).jpg",
    "/path/to/PMI(67).jpg", "/path/to/PMI(68).jpg", "/path/to/PMI(69).jpg", "/path/to/PMI(70).jpg", "/path/to/PMI(71).jpg", "/path/to/PMI(72).jpg", "/path/to/PMI(73).jpg", "/path/to/PMI(74).jpg", "/path/to/PMI(75).jpg", "/path/to/PMI(76).jpg", "/path/to/PMI(77).jpg", "/path/to/PMI(78).jpg", "/path/to/PMI(79).jpg", "/path/to/PMI(80).jpg", "/path/to/PMI(81).jpg", "/path/to/PMI(82).jpg", "/path/to/PMI(83).jpg", "/path/to/PMI(84).jpg", "/path/to/PMI(85).jpg", "/path/to/PMI(86).jpg", "/path/to/PMI(87).jpg", "/path/to/PMI(88).jpg", 
    "/path/to/PMI(89).jpg", "/path/to/PMI(90).jpg", "/path/to/PMI(91).jpg", "/path/to/PMI(92).jpg", "/path/to/PMI(93).jpg", "/path/to/PMI(94).jpg", "/path/to/PMI(95).jpg", "/path/to/PMI(96).jpg", "/path/to/PMI(97).jpg", "/path/to/PMI(98).jpg", "/path/to/PMI(99).jpg", "/path/to/PMI(100).jpg", "/path/to/PMI(101).jpg", "/path/to/PMI(102).jpg", "/path/to/PMI(103).jpg", "/path/to/PMI(104).jpg", "/path/to/PMI(105).jpg", "/path/to/PMI(106).jpg", "/path/to/PMI(107).jpg", "/path/to/PMI(108).jpg", "/path/to/PMI(109).jpg", "/path/to/PMI(110).jpg", 
    "/path/to/PMI(111).jpg", "/path/to/PMI(112).jpg", "/path/to/PMI(113).jpg", "/path/to/PMI(114).jpg", "/path/to/PMI(115).jpg", "/path/to/PMI(116).jpg", "/path/to/PMI(117).jpg", "/path/to/PMI(118).jpg", "/path/to/PMI(119).jpg", "/path/to/PMI(120).jpg"],

    "Normal": ["/path/to/Normal(1).jpg", "/path/to/Normal(2).jpg", "/path/to/Normal(3).jpg", "/path/to/Normal(4).jpg", "/path/to/Normal(5).jpg", "/path/to/Normal(6).jpg", "/path/to/Normal(7).jpg", "/path/to/Normal(8).jpg", "/path/to/Normal(9).jpg", "/path/to/Normal(10).jpg", "/path/to/Normal(11).jpg", "/path/to/Normal(12).jpg", "/path/to/Normal(13).jpg", "/path/to/Normal(14).jpg", "/path/to/Normal(15).jpg", "/path/to/Normal(16).jpg", "/path/to/Normal(17).jpg", "/path/to/Normal(18).jpg", "/path/to/Normal(19).jpg", "/path/to/Normal(20).jpg", "/path/to/Normal(21).jpg", "/path/to/Normal(22).jpg",
    "/path/to/Normal(23).jpg", "/path/to/Normal(24).jpg", "/path/to/Normal(25).jpg", "/path/to/Normal(26).jpg", "/path/to/Normal(27).jpg", "/path/to/Normal(28).jpg", "/path/to/Normal(29).jpg", "/path/to/Normal(30).jpg", "/path/to/Normal(31).jpg", "/path/to/Normal(32).jpg", "/path/to/Normal(33).jpg", "/path/to/Normal(34).jpg", "/path/to/Normal(35).jpg", "/path/to/Normal(36).jpg", "/path/to/Normal(37).jpg", "/path/to/Normal(38).jpg", "/path/to/Normal(39).jpg", "/path/to/Normal(40).jpg", "/path/to/Normal(41).jpg", "/path/to/Normal(42).jpg", "/path/to/Normal(43).jpg", "/path/to/Normal(44).jpg",
    "/path/to/Normal(45).jpg", "/path/to/Normal(46).jpg", "/path/to/Normal(47).jpg", "/path/to/Normal(48).jpg", "/path/to/Normal(49).jpg", "/path/to/Normal(50).jpg", "/path/to/Normal(51).jpg", "/path/to/Normal(52).jpg", "/path/to/Normal(53).jpg", "/path/to/Normal(54).jpg", "/path/to/Normal(55).jpg", "/path/to/Normal(56).jpg", "/path/to/Normal(57).jpg", "/path/to/Normal(58).jpg", "/path/to/Normal(59).jpg", "/path/to/Normal(60).jpg", "/path/to/Normal(61).jpg", "/path/to/Normal(62).jpg", "/path/to/Normal(63).jpg", "/path/to/Normal(64).jpg", "/path/to/Normal(65).jpg", "/path/to/Normal(66).jpg", 
    "/path/to/Normal(67).jpg", "/path/to/Normal(68).jpg", "/path/to/Normal(69).jpg", "/path/to/Normal(70).jpg", "/path/to/Normal(71).jpg", "/path/to/Normal(72).jpg", "/path/to/Normal(73).jpg", "/path/to/Normal(74).jpg", "/path/to/Normal(75).jpg", "/path/to/Normal(76).jpg", "/path/to/Normal(77).jpg", "/path/to/Normal(78).jpg", "/path/to/Normal(79).jpg", "/path/to/Normal(80).jpg", "/path/to/Normal(81).jpg", "/path/to/Normal(82).jpg", "/path/to/Normal(83).jpg", "/path/to/Normal(84).jpg", "/path/to/Normal(85).jpg", "/path/to/Normal(86).jpg", "/path/to/Normal(87).jpg", "/path/to/Normal(88).jpg", 
    "/path/to/Normal(89).jpg", "/path/to/Normal(90).jpg", "/path/to/Normal(91).jpg", "/path/to/Normal(92).jpg", "/path/to/Normal(93).jpg", "/path/to/Normal(94).jpg", "/path/to/Normal(95).jpg", "/path/to/Normal(96).jpg", "/path/to/Normal(97).jpg", "/path/to/Normal(98).jpg", "/path/to/Normal(99).jpg", "/path/to/Normal(100).jpg", "/path/to/Normal(101).jpg", "/path/to/Normal(102).jpg", "/path/to/Normal(103).jpg", "/path/to/Normal(104).jpg", "/path/to/Normal(105).jpg", "/path/to/Normal(106).jpg", "/path/to/Normal(107).jpg", "/path/to/Normal(108).jpg", "/path/to/Normal(109).jpg", "/path/to/Normal(110).jpg",
    "/path/to/Normal(111).jpg", "/path/to/Normal(112).jpg", "/path/to/Normal(113).jpg", "/path/to/Normal(114).jpg", "/path/to/Normal(115).jpg", "/path/to/Normal(116).jpg", "/path/to/Normal(117).jpg", "/path/to/Normal(118).jpg", "/path/to/Normal(119).jpg", "/path/to/Normal(120).jpg"],
  }
  organizeimages(images, names, imagepathsforallclasses)
def call():
  return imagesinclass
  
