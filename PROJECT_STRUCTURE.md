## Project Structure

The PyVerse repository is organized as follows:

<!-- START_STRUCTURE -->
```
├── Advanced_Projects
├── Algorithms_and_Data_Structures
│   ├── Linked List
│   │   ├── Menu_Driven_Code_for_Circular_Doubly_LinkedList.py
│   │   ├── Menu_Driven_Code_for_Circular_LinkedList.py
│   │   ├── Menu_Driven_Code_for_Doubly_LinkedList.py
│   │   ├── Menu_Driven_Code_for_Dynamic_Linear_Queue_using_LinkedList.py
│   │   ├── Menu_Driven_Code_for_Dynamic_Stack_using_LinkedList.py
│   │   ├── Menu_Driven_Code_for_Linear_LinkedList.py
│   │   └── README.md
│   ├── Stack
│   │   ├── README.md
│   │   └── stack.py
│   └── Trees
│       ├── Menu_Driven_Code_for_Avl_Tree.py
│       ├── Menu_Driven_Code_for_Binary_Search_Tree.py
│       ├── Menu_Driven_Code_for_Binary_Tree.py
│       ├── Menu_Driven_Code_for_DFS.py
│       ├── Menu_Driven_Code_for_Tree_Traversals.py
│       └── README.md
├── Automation_Tools
├── Beginner_Projects
│   ├── Calculator_App
│   │   ├── README.md
│   │   └── main.py
│   ├── Morse Code Translator with GUI
│   │   ├── README.md
│   │   ├── main.py
│   │   └── screenshots
│   │       └── tkinter-working.gif
│   ├── QR Generator
│   │   ├── README.md
│   │   └── generate_qrcode.py
│   ├── Stock App
│   │   ├── Readme.md
│   │   ├── Templates
│   │   │   ├── base.html
│   │   │   ├── financials.html
│   │   │   └── index.html
│   │   └── server.py
│   └── Turtle
│       ├── Readme.md
│       ├── rainbow_spiral.py
│       └── turtle_spiral.py
├── Blockchain_Development
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Cybersecurity_Tools
│   └── CLI-based Port Scanner
│       ├── README.md
│       └── port-scanner.py
├── DataVizLearnig
│   ├── DataViz_Snippets.ipynb
│   └── Readme.md
├── Data_Science
│   ├── Data-science.md
│   └── time_series_visualization
│       ├── README.md
│       ├── Time_Series_Report.pdf
│       ├── Time_Series_Visualization.ipynb
│       ├── airline_passengers.csv
│       ├── autocorrelation_plot.png
│       ├── eda_plot.png
│       ├── exponential_smoothing_plot.png
│       ├── moving_average_plot.png
│       ├── seasonal_plot.png
│       └── trend_analysis_plot.png
├── Deep_Learning
│   ├── Bird Species Classification
│   │   ├── Dataset
│   │   │   └── Readme.md
│   │   ├── Images
│   │   │   ├── InceptionV3.png
│   │   │   ├── inception_resnet_v2 .png
│   │   │   ├── masked_image_1.png
│   │   │   ├── masked_image_2.png
│   │   │   └── masked_image_3.png
│   │   ├── Model
│   │   │   └── bird_species_classification.ipynb
│   │   └── Readme.md
│   ├── Face Mask Detection
│   │   ├── Dataset
│   │   │   └── Readme.md
│   │   ├── Images
│   │   │   ├── Distribution of classes.jpg
│   │   │   ├── Evaluation.jpg
│   │   │   ├── Readme.md
│   │   │   └── Sample Images.jpg
│   │   ├── Model
│   │   │   ├── Readme.md
│   │   │   └── detecting-face-masks-with-5-models.ipynb
│   │   └── requirements.txt
│   ├── MNIST Digit Classification using Neural Networks
│   │   ├── README.md
│   │   ├── bar graph.png
│   │   ├── dataset
│   │   │   └── readme.md
│   │   ├── histogram.png
│   │   ├── images
│   │   │   ├── bar graph.png
│   │   │   ├── confusion matrix.png
│   │   │   ├── histogram.png
│   │   │   ├── input visualisation.png
│   │   │   ├── pie chart.png
│   │   │   └── training loss.png
│   │   ├── input visualisation.png
│   │   ├── model
│   │   │   ├── ANN_Handwritten_Digit_Classification.ipynb
│   │   │   └── CNN_handwritten_digit_recogniser.ipynb
│   │   ├── pie chart.png
│   │   └── requirement.txt
│   ├── Plant Disease Detection
│   │   ├── Final tensorflow Models
│   │   │   ├── cotton.h5
│   │   │   ├── cucumber.h5
│   │   │   ├── grapes.h5
│   │   │   ├── guava.h5
│   │   │   ├── potato.h5
│   │   │   ├── rice.h5
│   │   │   ├── sugarcane.h5
│   │   │   ├── tomato.h5
│   │   │   └── wheat.h5
│   │   ├── README.md
│   │   ├── assets
│   │   │   └── images
│   │   │       ├── cotton_result-graph.png
│   │   │       ├── cotton_result.png
│   │   │       ├── grapes_result.png
│   │   │       ├── grapes_result_graph.png
│   │   │       ├── guava_result.png
│   │   │       ├── guava_result_graph.png
│   │   │       ├── potato_result_graph.png
│   │   │       ├── sugarcane_result_graph.png
│   │   │       └── tomato_result_graph.png
│   │   ├── ipynb files
│   │   │   ├── Cotton_Classification.ipynb
│   │   │   ├── Grapes_Classification.ipynb
│   │   │   ├── Guava_Classification.ipynb
│   │   │   ├── Potato_Classification.ipynb
│   │   │   ├── Sugarcane_Classification.ipynb
│   │   │   ├── Tomato_Classification.ipynb
│   │   │   └── requirements.txt
│   │   └── result.md
│   └── Spam Vs Ham Mail Classification [With Streamlit GUI]
│       ├── Dataset
│       │   ├── newData.csv
│       │   └── spam-vs-ham-dataset.csv
│       ├── Image
│       │   ├── PairPlot_withHue.png
│       │   ├── Spam-vs-ham-piechart.jpg
│       │   ├── spam-ham-num_chr.jpg
│       │   ├── spam-ham-num_sent.jpg
│       │   └── spam-ham-num_word.jpg
│       ├── Model
│       │   ├── README.md
│       │   ├── app1.py
│       │   ├── app2.py
│       │   ├── model1.ipynb
│       │   └── model2.ipynb
│       └── requirements.txt
├── Game_Development
├── LICENSE
├── Machine_Learning
│   ├── Air Quality Prediction
│   │   ├── Dataset
│   │   │   └── README.md
│   │   ├── Images
│   │   │   ├── Satisfaction_level_of_people_post_covid.jpg
│   │  