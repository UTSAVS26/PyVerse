from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery
def get_movie_details(movie_name):

    # Load the RDF data
    g = Graph()
    g.parse("data/Movies.rdf")

    # Define namespaces
    rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    ont = Namespace("http://www.semanticweb.org/owl/owlapi/turtle#")

    # Query RDF graph for movie information
    query_text = """
    SELECT ?year ?country ?genre ?actor
    WHERE {
        ?movie rdf:type ont:Movie ;
               rdfs:label ?title .
        OPTIONAL { ?movie ont:hasYear ?year }
        OPTIONAL { ?movie ont:hasCountry ?country }
        OPTIONAL { ?movie ont:hasGenre ?genre }
        OPTIONAL { ?movie ont:hasActor ?actor }
        FILTER (?title = "%s")
    }

    """ % movie_name

    query = prepareQuery(query_text, initNs={"rdf": rdf, "ont": ont , "rdfs": rdfs})
    results = g.query(query)

    # Check if any results are found
    if len(results) == 0:
        return None
    else:
        # Initialize lists to store genres and actors
        genres = []
        actors = []
        year = None
        country = None

        # Iterate through results and accumulate genres and actors
        for row in results:
            _, _, genre, actor = row
            if genre and genre.split("#")[1] not in genres:
                genres.append(genre.split("#")[1])
            if actor and actor.split("#")[1] not in actors:
                actors.append(actor.split("#")[1])
            if row.year and not year:
                year = row.year
            if row.country and not country:
                country = row.country

        # Print movie details
        # print("Year:", year if year else "Not available")
        # print("Country:", country if country else "Not available")
        # print("Genres:", ', '.join(genres) if genres else "Not available")
        # print("Actors:", ', '.join(actors) if actors else "Not available")
        details = []
        details.append("Year: " + (year if year else "Not available"))
        details.append("Country: " + (country if country else "Not available"))
        details.append("Genres: " + (', '.join(genres) if genres else "Not available"))
        # details.append("Actors: " + (', '.join(actors) if actors else "Not available"))
        details.append("Actors: " + (', '.join(actor.replace('_', ' ') for actor in actors) if actors else "Not available"))
        return details
