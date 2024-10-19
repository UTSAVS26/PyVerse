from rdflib import Graph, Namespace

def get_movies(file_path = "data/Movies.rdf", included_actors=None, excluded_actors=None,
               included_directors=None, excluded_directors=None,
               included_genres=None, excluded_genres=None):
    # Load RDF file
    g = Graph()
    g.parse(file_path)

    # Define namespaces
    onto = Namespace("http://www.semanticweb.org/owl/owlapi/turtle#")

    # Constructing SPARQL query
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX onto: <http://www.semanticweb.org/owl/owlapi/turtle#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX inst: <http://www.co-ode.org/ontologies/ont.owl#>
    SELECT ?title
    WHERE {
        ?movie rdf:type onto:Movie ;
               rdfs:label ?title .
    """

    # Filter by included actors
    if included_actors:
        for actor in included_actors:
            query += f"""
        ?movie onto:hasActor inst:{actor} .
            """

    # Filter by excluded actors
    if excluded_actors:
        for actor in excluded_actors:
            query += f"""
        FILTER NOT EXISTS {{ ?movie onto:hasActor inst:{actor} . }}
            """

    # Filter by included directors
    if included_directors:
        for director in included_directors:
            query += f"""
        ?movie onto:hasDirector inst:{director} .
            """

    # Filter by excluded directors
    if excluded_directors:
        for director in excluded_directors:
            query += f"""
        FILTER NOT EXISTS {{ ?movie onto:hasDirector inst:{director} . }}
            """

    # Filter by included genres
    if included_genres:
        for genre in included_genres:
            query += f"""
        ?movie onto:hasGenre inst:{genre} .
            """

    # Filter by excluded genres
    if excluded_genres:
        for genre in excluded_genres:
            query += f"""
        FILTER NOT EXISTS {{ ?movie onto:hasGenre inst:{genre} . }}
            """

    query += "}\n"  # Closing query brace

    # Execute the SPARQL query
    results = g.query(query)

    return results

# Example usage:
file_path = "data/Movies.rdf"
included_actors = ["Uma_Thurman", "John_Travolta"]
excluded_actors = []
included_directors = []
excluded_directors = []
included_genres = ["Thriller"]
excluded_genres = []
results = get_movies(file_path, included_actors=included_actors, excluded_actors=None,
               included_directors=None, excluded_directors=None,
               included_genres=included_genres, excluded_genres=None)
if results:
    print("Movies found:")
    for movie in results:
        print(movie[0].value)
else:
    print("No movies found.")

