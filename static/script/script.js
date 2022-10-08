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
