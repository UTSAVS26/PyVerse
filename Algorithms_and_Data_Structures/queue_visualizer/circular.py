import streamlit as st

class CircularQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = [None] * max_size
        self.front = self.rear = -1

    def is_empty(self):
        return self.front == -1

    def is_full(self):
        return (self.rear + 1) % self.max_size == self.front

    def enqueue(self, item):
        if self.is_full():
            st.warning("Queue is full. Cannot enqueue.")
        elif self.is_empty():
            self.front = self.rear = 0
            self.queue[self.rear] = item
        else:
            self.rear = (self.rear + 1) % self.max_size
            self.queue[self.rear] = item

    def dequeue(self):
        if self.is_empty():
            st.warning("Queue is empty. Cannot dequeue.")
            return None
        elif self.front == self.rear:
            item = self.queue[self.front]
            self.front = self.rear = -1
            return item
        else:
            item = self.queue[self.front]
            self.front = (self.front + 1) % self.max_size
            return item

    def display_queue(self):
        return self.queue

# Helper function to display the queue in a box-like format
def display_queue_as_boxes(queue, front, rear, max_size):
    box_representation = ""
    for i in range(max_size):
        if i == front and i == rear and queue[i] is not None:
            # Both front and rear pointing to the same item (single item)
            box_representation += f"| **Front/Rear** {queue[i]} |"
        elif i == front:
            box_representation += f"| **Front** {queue[i]} |"
        elif i == rear:
            box_representation += f"| **Rear** {queue[i]} |"
        else:
            box_representation += f"| {queue[i]} |"
    return box_representation

# Streamlit UI
st.title("Circular Queue Operations with Box Display")

# Queue size input
max_size = st.number_input("Enter the size of the circular queue:", min_value=3, max_value=10, value=5, step=1)
cq = CircularQueue(max_size)

# Initialize session state for queue
if "queue_state" not in st.session_state:
    st.session_state.queue_state = cq

# Options to enqueue, dequeue, and display
option = st.selectbox("Choose an operation", ("None", "Enqueue", "Dequeue", "Display"))

if option == "Enqueue":
    item = st.number_input("Enter item to enqueue:", min_value=0, value=1)
    if st.button("Enqueue Item"):
        st.session_state.queue_state.enqueue(item)
        st.success(f"Enqueued {item}")
        queue = st.session_state.queue_state.display_queue()
        st.markdown(display_queue_as_boxes(queue, st.session_state.queue_state.front, st.session_state.queue_state.rear, max_size))

if option == "Dequeue":
    if st.button("Dequeue Item"):
        dequeued_item = st.session_state.queue_state.dequeue()
        if dequeued_item is not None:
            st.success(f"Dequeued {dequeued_item}")
        queue = st.session_state.queue_state.display_queue()
        st.markdown(display_queue_as_boxes(queue, st.session_state.queue_state.front, st.session_state.queue_state.rear, max_size))

if option == "Display":
    if st.button("Show Queue"):
        queue = st.session_state.queue_state.display_queue()
        if any(queue):
            st.markdown(display_queue_as_boxes(queue, st.session_state.queue_state.front, st.session_state.queue_state.rear, max_size))
        else:
            st.warning("Queue is empty.")

