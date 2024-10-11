<?php
// Establish database connection
$conn = new mysqli("127.0.0.1", "root", "", "restaurantdetails");

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Function to check if the user can book a table
function canBookTable($conn) {
    $sixHoursAgo = date('Y-m-d H:i:s', strtotime('-6 hours'));
    $sql = "SELECT * FROM bookings WHERE phone = ? AND booking_time > ?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("ss", $_POST['phone'], $sixHoursAgo);
    $stmt->execute();
    $result = $stmt->get_result();
    return $result->num_rows == 0;
}

// Function to assign a random table number
function assignTableNumber() {
    return rand(1, 30);
}

// Handle form submission
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Check if user can book a table
    if (!canBookTable($conn)) {
        echo "You have already booked a table within the last 6 hours.";
        exit;
    }

    // Assign table number
    $tableNumber = assignTableNumber();

    // Insert booking into the database
    $sql = "INSERT INTO bookings (name, phone, table_number) VALUES (?, ?, ?)";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("ssi", $_POST['name'], $_POST['phone'], $tableNumber);
    if ($stmt->execute()) {
        echo "Booking successful! Your table number is: $tableNumber";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

$conn->close();
?>
