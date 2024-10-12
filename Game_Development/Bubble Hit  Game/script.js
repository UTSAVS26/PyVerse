let timer = 60; // Initialize timer with 60 seconds
let score = 0; // Initialize score to 0
let random_num_hit = 0; // Initialize the number the player has to click on

// Function to increase the player's score by 10
function increase_score() {
   score += 10; // Increment score by 10
   document.querySelector("#scorevalue").textContent = score; // Update score display in the UI
}

// Function to generate bubbles with random numbers
function make_bubble() {
   let bubble_counter = ""; // Initialize an empty string for the bubbles

   // Loop to create 102 bubbles with random numbers between 0 and 9
   for (let i = 1; i <= 102; i++) {
      let random_num = Math.floor(Math.random() * 10); // Generate random number between 0 and 9
      bubble_counter += `<div class="bubble"> ${random_num} </div>`; // Add a bubble with the random number
   }

   // Add the generated bubbles to the game area (element with id 'pbtm')
   document.querySelector("#pbtm").innerHTML = bubble_counter;
}

// Function to generate a new random number for the player to hit
function new_hit() {
   random_num_hit = Math.floor(Math.random() * 10); // Generate new random number between 0 and 9
   document.querySelector("#hitvalue").textContent = random_num_hit; // Update the hit value in the UI
}

// Function to start and manage the countdown timer
function run_timer() {
   let timerint = setInterval(function () {
      if (timer > 0) {
         timer--; // Decrease timer by 1 second
         document.querySelector("#timervalue").textContent = timer; // Update timer display in the UI
      } else {
         clearInterval(timerint); // Stop the timer when it reaches 0
         document.querySelector("#pbtm").innerHTML = `<h1> <i>Game Over</i><h1>`; // Display "Game Over"
      }
   }, 1000); // Run every second (1000 ms)
}

// Event listener for clicks on the bubbles
document.querySelector("#pbtm").addEventListener("click", function (bubble) {
   let clickedNumber = Number(bubble.target.textContent); // Get the number from the clicked bubble

   // Check if the clicked number matches the target number
   if (clickedNumber === random_num_hit) {
      increase_score(); // Increase the player's score
      make_bubble(); // Regenerate bubbles with new random numbers
      new_hit(); // Generate a new target number
   }
});

// Start the game
run_timer(); // Start the countdown timer
make_bubble(); // Generate initial bubbles
new_hit(); // Generate the first target number
