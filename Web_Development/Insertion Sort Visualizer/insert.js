const wrapper = document.querySelector(".main-container"),
input_box = wrapper.querySelector("#input-box");
let addBtn = document.getElementById("add-btn");
let sortBtn = document.getElementById("sort-btn");
let resetBtn = document.getElementById("reset-btn");
let input_section = document.querySelector(".input-section");
let output_section = document.querySelector(".output-section");
let array = [];
let step_containerNo = 0;

function addNumber() {
  let getData = input_box.value;
  getData = parseInt(getData);
  if (!Number.isNaN(getData)) {
    let box = document.createElement("div");
    box.appendChild(document.createTextNode(getData));
    box.setAttribute("class", "box");
    document.querySelector(".input-section").appendChild(box);
    array.push(getData);
  }
  console.log(array);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

sortBtn.addEventListener("click", async () => {
  for (let i = 1; i < array.length; i++) {
    show_Array(array);
    await sleep(500);
    let box =
      output_section.querySelectorAll(".step_container")[step_containerNo];
    let array_element = box.getElementsByClassName("box");
    let key = parseInt(array_element[i].textContent);
    await showKey(key);
    array_element[i].classList.add("key-element");
    await Target_leftArray(i, array_element, true);
    await sleep(1000);
    let j = i - 1;

    while (j >= 0 && parseInt(array_element[j].textContent) > key) {
      await swap(array_element[j], array_element[j + 1]);
      array[j + 1] = array[j];
      array_element[j].style.backgroundColor = "antiquewhite";
      await sleep(2000);
      j = j - 1;
    }

    let box1 =
      output_section.querySelectorAll(".step_container")[step_containerNo];
    let key_container = box1.querySelector(".key-section");
    let element = key_container.querySelector(".box");
    element.style.position = "relative";
    await animate1(element);
    await insertKey(element, j + 1, key);
    array[j + 1] = key;
    await Target_leftArray(i, array_element, false);
    array_element[i].classList.remove("key-element");
    step_containerNo += 1;
  }
  console.log(array);
});

function show_Array(array) {
  let step_container = document.createElement("div");
  step_container.setAttribute("class", "step_container");
  for (let i = 0; i < array.length; i++) {
    let box = document.createElement("div");
    box.appendChild(document.createTextNode(array[i]));
    box.setAttribute("class", "box");
    step_container.appendChild(box);
  }
  output_section.appendChild(step_container);
}
function animate1(element) {
  return new Promise((resolve) => {
    element.animate(
      {
        top: "100px",
      },
      1000
    );
    setTimeout(() => {
      resolve();
    }, 1000);
  });
}

function insertKey(element, pos, key) {
  let box =
    output_section.querySelectorAll(".step_container")[step_containerNo];
  let array_element = box.getElementsByClassName("box");
  let key_container = box.querySelector(".key-section");
  element.style.top = "100px";
  return new Promise((resolve) => {
    element.animate(
      {
        left: `${
          $(array_element[pos]).offset().left - $(element).offset().left
        }px`,
      },
      1000
    );
    element.style.left = `${
      $(array_element[pos]).offset().left - $(element).offset().left
    }px`;
    setTimeout(() => {
      array_element[pos].textContent = key;
      array_element[pos].style.backgroundColor = "purple";
      key_container.querySelector("i").classList.add("animate-key");
      key_container.style.display = "none";
      resolve();
    }, 1000);
  });
}

async function showKey(key) {
  return new Promise((resolve) => {
    setTimeout(() => {
      let box =
        output_section.querySelectorAll(".step_container")[step_containerNo];
      const node = document.createElement("div");
      node.setAttribute("class", "key-section");
      const keynode = document.createElement("div");
      keynode.classList.add("box");
      keynode.innerHTML = key;
      node.appendChild(keynode);
      let arrow_box = document.createElement("div");
      arrow_box.innerHTML = `<i class="fa fa-arrow-left" aria-hidden="true"></i>`;
      node.appendChild(arrow_box);
      arrow_box.appendChild(document.createTextNode("Key"));
      box.appendChild(node);
      resolve();
    }, 500);
  });
}

async function Target_leftArray(index, array_element, situation) {
  return new Promise((resolve) => {
    setTimeout(() => {
      for (let i = 0; i < index; i++) {
        if (situation === true) {
          array_element[i].classList.add("target_elements");
        } else {
          array_element[i].classList.remove("target_elements");
        }
      }
      resolve();
    }, 1000);
  });
}

function swap(element, target) {
  element.style.backgroundColor = "#e6852c";
  if (target.innerHTML != "") target.style.backgroundColor = "#e6852c";
  let d = element.cloneNode(true);
  d.animate(
    {
      transform: [
        `translate(${$(target).offset().left - $(element).offset().left}px)`,
      ],
    },
    1000
  );
  return new Promise((resolve) => {
    setTimeout(() => {
      target.innerHTML = element.innerHTML;
      target.style.backgroundColor = "purple";
      element.style.backgroundColor = "antiquewhite";
      element.style.boxShadow = "none";
      element.innerHTML = "";
      resolve();
    }, 1000);
  });
}

function resetData() {
  input_section.innerHTML = "";
  output_section.innerHTML = "";
}
