var Point = 1;
var X1, Y1, X2, Y2;

function FindPosition(oElement) {
  if (typeof oElement.offsetParent != "undefined") {
    for (var posX = 0, posY = 0; oElement; oElement = oElement.offsetParent) {
      posX += oElement.offsetLeft;
      posY += oElement.offsetTop;
    }
    return [posX, posY];
  } else {
    return [oElement.x, oElement.y];
  }
}

function GetCoordinates(e) {
  var PosX = 0;
  var PosY = 0;
  var ImgPos;
  ImgPos = FindPosition(myImg);
  if (!e) var e = window.event;
  if (e.pageX || e.pageY) {
    PosX = e.pageX;
    PosY = e.pageY;
  } else if (e.clientX || e.clientY) {
    PosX =
      e.clientX +
      document.body.scrollLeft +
      document.documentElement.scrollLeft;
    PosY =
      e.clientY + document.body.scrollTop + document.documentElement.scrollTop;
  }
  PosX = PosX - ImgPos[0];
  PosY = PosY - ImgPos[1];
  if (Point == 1) {
    X1 = PosX;
    Y1 = PosY;
    Point = 2;
    document.getElementById("x1").innerHTML = PosX;
    document.getElementById("y1").innerHTML = PosY;
  } else {
    X2 = PosX;
    Y2 = PosY;
    Point = 3;
    document.getElementById("x2").innerHTML = PosX;
    document.getElementById("y2").innerHTML = PosY;
    document.form1.drawbutton.disabled = false;
  }
}

function Clear() {
  Point = 1;
  document.getElementById("x1").innerHTML = "";
  document.getElementById("y1").innerHTML = "";
  document.getElementById("x2").innerHTML = "";
  document.getElementById("y2").innerHTML = "";
  document.form1.drawbutton.disabled = true;
  myImg.src = "rectangle.asp";
}

function Draw() {
  myImg.src =
    "rectangle.asp?x1=" + X1 + "&y1=" + Y1 + "&x2=" + X2 + "&y2=" + Y2;
  document.form1.drawbutton.disabled = true;
}

function Initialisation() {
  document.form1.drawbutton.disabled = true;
}
