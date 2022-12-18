var actuallog = console.log;

console.log = (...args) => {
  args.map(
    (arg) => (document.querySelector("#mylog").innerHTML += arg + "<br>")
  );
};

// substitution test
document.getElementById("embed").innerHTML = "{{embed}}";

// fetch json
fetch("/test")
  .then(function (response) {
    return response.json();
  })
  .then(function (text) {
    console.log("GET response text:");
    console.log(text); // Print the greeting as text
    console.log(JSON.stringify(text));
    console.log("");
  });

// fetch results
fetch("/getdata/10/33")
  .then(function (response) {
    return response.text();
  })
  .then(function (text) {
    console.log("GET response text:");
    console.log(text); // Print the greeting as text
  });

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
  document.getElementById("x_coord").value = PosX;
  document.getElementById("y_coord").value = PosY;
}

function sliderChange(val) {
  document.getElementById("output").innerHTML = val; // get
}
