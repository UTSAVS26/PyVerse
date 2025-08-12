from graphviz import Digraph

def generate_dag_diagram():
    dag = Digraph(format='png')
    dag.attr(rankdir='LR', size='10')

    # Nodes
    dag.node("Start", shape="oval")
    dag.node("InputTopic", "UserInput: Topic", shape="box")
    dag.node("A", "Agent A: Scientist", shape="box")
    dag.node("B", "Agent B: Philosopher", shape="box")
    dag.node("M", "Memory Node", shape="parallelogram")
    dag.node("J", "Judge Node", shape="diamond")
    dag.node("End", shape="oval")

    # Edges
    dag.edge("Start", "InputTopic")
    dag.edge("InputTopic", "A")
    dag.edge("A", "M")
    dag.edge("M", "B")
    dag.edge("B", "M")
    dag.edge("M", "A", label="x4 alternating")
    dag.edge("M", "J")
    dag.edge("J", "End")

    dag.render("debate_dag_diagram", format="png", cleanup=True)
    print("DAG diagram generated and saved as 'debate_dag_diagram.png'")

if __name__ == "__main__":
    generate_dag_diagram()