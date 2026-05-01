import cv2
from deepface import DeepFace
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------
# STEP 1: LOAD IMAGE
# -------------------------------
image_path = "test4.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

# -------------------------------
# STEP 2: FACE ANALYSIS
# -------------------------------
results = DeepFace.analyze(
    img,
    actions=["age", "gender"],
    detector_backend="retinaface",
    enforce_detection=True
)

if isinstance(results, dict):
    results = [results]

people = []

for i, face in enumerate(results):
    gender_scores = face.get("gender", {})
    dominant_gender = face.get("dominant_gender")

    confidence = float(gender_scores.get(dominant_gender, 0.0)) if dominant_gender else 0.0

    person = {
        "person_id": i + 1,
        "age": int(round(float(face.get("age", 0)))),
        "gender": dominant_gender,
        "gender_confidence": round(confidence, 2)
    }

    people.append(person)

# -------------------------------
# STEP 3: SMART RULE ENGINE
# -------------------------------
def predict_relationship(p1, p2):
    age1, age2 = p1["age"], p2["age"]
    g1, g2 = p1["gender"], p2["gender"]

    age_diff = abs(age1 - age2)

    def get_role(age):
        if age < 14:
            return "child"
        elif age < 45:
            return "adult"
        else:
            return "elder"

    r1, r2 = get_role(age1), get_role(age2)

    # Grandparent
    if ("elder" in [r1, r2]) and ("child" in [r1, r2]):
        return "grandparent-child"

    # Parent-child
    if ("adult" in [r1, r2]) and ("child" in [r1, r2]):
        if g1 == "Woman" or g2 == "Woman":
            return "mother-child"
        else:
            return "father-child"

    # Adult + Elder
    if ("adult" in [r1, r2]) and ("elder" in [r1, r2]):
        return "parent-grandparent"

    # Same age group
    if r1 == r2:
        if r1 == "child":
            return "siblings"
        if r1 == "adult":
            return "friends/colleagues"
        if r1 == "elder":
            return "elderly relatives"

    return "unknown"

# -------------------------------
# STEP 4: PRINT RESULTS
# -------------------------------
print("Detected People:")
for p in people:
    print(p)

print("\nPredicted Relationships:")

relationships = []

for i in range(len(people)):
    for j in range(i + 1, len(people)):
        relation = predict_relationship(people[i], people[j])

        pair = (people[i]["person_id"], people[j]["person_id"])

        relationships.append((pair[0], pair[1], relation))

        print({
            "pair": pair,
            "relation": relation
        })

# -------------------------------
# STEP 5: GRAPH VISUALIZATION
# -------------------------------
G = nx.Graph()

# Add nodes
for p in people:
    label = f"P{p['person_id']}\nAge:{p['age']}"
    G.add_node(p["person_id"], label=label)

# Add edges
for r in relationships:
    G.add_edge(r[0], r[1], label=r[2])

# Draw graph
pos = nx.spring_layout(G)

labels = nx.get_node_attributes(G, 'label')
edge_labels = nx.get_edge_attributes(G, 'label')

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Social Relationship Graph")
plt.show()