import streamlit as st

# Helper function to display the queue in a box-like format
def display_queue_as_boxes(queue, front=None, rear=None):
    box_representation = ""
    for i, item in enumerate(queue):
        label = ""
        if i == front:
            label = "**Front**"
        elif i == rear:
            label = "**Rear**"
        box_representation += f"| {label} {item} |"
    return box_representation

### Linear Queue ###
class LinearQueue:
    def __init__(self, max_size):
        self.queue = []
        self.max_size = max_size

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.max_size

    def enqueue(self, item):
        if self.is_full():
            st.warning("Linear queue is full.")
        else:
            self.queue.append(item)

    def dequeue(self):
        if self.is_empty():
            st.warning("Linear queue is empty.")
        else:
            return self.queue.pop(0)

    def display_queue(self):
        return self.queue

### Dequeue (Double-ended Queue) ###
class Dequeue:
    def __init__(self, max_size):
        self.queue = []
        self.max_size = max_size

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.max_size

    def add_front(self, item):
        if self.is_full():
            st.warning("Dequeue is full.")
        else:
            self.queue.insert(0, item)

    def add_rear(self, item):
        if self.is_full():
            st.warning("Dequeue is full.")
        else:
            self.queue.append(item)

    def remove_front(self):
        if self.is_empty():
            st.warning("Dequeue is empty.")
        else:
            return self.queue.pop(0)

    def remove_rear(self):
        if self.is_empty():
            st.warning("Dequeue is empty.")
        else:
            return self.queue.pop()

    def display_queue(self):
        return self.queue

### Priority Queue ###
class PriorityQueue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)
        self.queue.sort()  # Sorts the queue after each insertion

    def dequeue(self):
        if len(self.queue) == 0:
            st.warning("Priority queue is empty.")
        else:
            return self.queue.pop(0)  # Pops the highest priority element

    def display_queue(self):
        return self.queue

### Queue Using Two Stacks ###
class QueueTwoStacks:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def enqueue(self, item):
        self.stack1.append(item)

    def dequeue(self):
        if len(self.stack2) == 0:
            if len(self.stack1) == 0:
                st.warning("Queue using two stacks is empty.")
            while len(self.stack1) > 0:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

    def display_queue(self):
        return self.stack2[::-1] + self.stack1  # Stack 2 reversed + Stack 1


# Streamlit UI
st.title("Different Types of Queues with Box-like Display")

# Queue type selection
queue_type = st.selectbox("Select the type of queue", ("Linear Queue", "Dequeue", "Priority Queue", "Queue Using Two Stacks"))

# Queue size input (for applicable queues)
if queue_type in ["Linear Queue", "Dequeue"]:
    max_size = st.number_input("Enter the size of the queue:", min_value=3, max_value=10, value=5, step=1)

# Initialize queues based on selection
if "queue_state" not in st.session_state:
    if queue_type == "Linear Queue":
        st.session_state.queue_state = LinearQueue(max_size)
    elif queue_type == "Dequeue":
        st.session_state.queue_state = Dequeue(max_size)
    elif queue_type == "Priority Queue":
        st.session_state.queue_state = PriorityQueue()
    elif queue_type == "Queue Using Two Stacks":
        st.session_state.queue_state = QueueTwoStacks()

# Operations based on the selected queue type
if queue_type == "Linear Queue":
    option = st.selectbox("Choose an operation", ("None", "Enqueue", "Dequeue", "Display"))
    
    if option == "Enqueue":
        item = st.number_input("Enter item to enqueue:", min_value=0, value=1)
        if st.button("Enqueue Item"):
            st.session_state.queue_state.enqueue(item)
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

    if option == "Dequeue":
        if st.button("Dequeue Item"):
            st.session_state.queue_state.dequeue()
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

    if option == "Display":
        if st.button("Show Queue"):
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

elif queue_type == "Dequeue":
    option = st.selectbox("Choose an operation", ("None", "Add to Front", "Add to Rear", "Remove from Front", "Remove from Rear", "Display"))
    
    if option == "Add to Front":
        item = st.number_input("Enter item to add to front:", min_value=0, value=1)
        if st.button("Add to Front"):
            st.session_state.queue_state.add_front(item)
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

    if option == "Add to Rear":
        item = st.number_input("Enter item to add to rear:", min_value=0, value=1)
        if st.button("Add to Rear"):
            st.session_state.queue_state.add_rear(item)
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

    if option == "Remove from Front":
        if st.button("Remove from Front"):
            st.session_state.queue_state.remove_front()
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

    if option == "Remove from Rear":
        if st.button("Remove from Rear"):
            st.session_state.queue_state.remove_rear()
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

    if option == "Display":
        if st.button("Show Queue"):
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

elif queue_type == "Priority Queue":
    option = st.selectbox("Choose an operation", ("None", "Enqueue", "Dequeue", "Display"))

    if option == "Enqueue":
        item = st.number_input("Enter item to enqueue:", min_value=0, value=1)
        if st.button("Enqueue Item"):
            st.session_state.queue_state.enqueue(item)
            st.write("Current Queue (sorted):", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

    if option == "Dequeue":
        if st.button("Dequeue Item"):
            st.session_state.queue_state.dequeue()
            st.write("Current Queue (sorted):", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

    if option == "Display":
        if st.button("Show Queue"):
            st.write("Current Queue (sorted):", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

elif queue_type == "Queue Using Two Stacks":
    option = st.selectbox("Choose an operation", ("None", "Enqueue", "Dequeue", "Display"))

    if option == "Enqueue":
        item = st.number_input("Enter item to enqueue:", min_value=0, value=1)
        if st.button("Enqueue Item"):
            st.session_state.queue_state.enqueue(item)
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

    if option == "Dequeue":
        if st.button("Dequeue Item"):
            st.session_state.queue_state.dequeue()
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))

    if option == "Display":
        if st.button("Show Queue"):
            st.write("Current Queue:", display_queue_as_boxes(st.session_state.queue_state.display_queue()))
