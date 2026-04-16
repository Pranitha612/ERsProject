import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------------------
# UI
# -------------------------------
st.markdown("""
<style>
body { background-color: #f4f4f6; }

.container {
    width: 70%;
    margin: auto;
    background: white;
    padding: 30px;
    margin-top: 40px;
    border-radius: 16px;
    box-shadow: 0px 10px 40px rgba(0,0,0,0.08);
}

.title {
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="title">Social Relationship Analysis</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    original_img = img.copy()

    # -------------------------------
    # ANALYSIS
    # -------------------------------
    results = DeepFace.analyze(
        img,
        actions=["age", "gender", "emotion"],
        detector_backend="retinaface",
        enforce_detection=False
    )

    if isinstance(results, dict):
        results = [results]

    people = []

    for i, face in enumerate(results):
        region = face.get("region", {})
        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

        people.append({
            "person_id": i + 1,
            "age": int(face["age"]),
            "gender": face["dominant_gender"],
            "emotion": face["dominant_emotion"],
            "box": (x, y, w, h),
            "face_area": w * h
        })

    # -------------------------------
    # ROLE ASSIGNMENT (FIX)
    # -------------------------------
    def assign_roles(people):
        sorted_by_size = sorted(people, key=lambda x: x["face_area"])

        if len(sorted_by_size) == 1:
            sorted_by_size[0]["role"] = "adult"
            return people

        sorted_by_size[0]["role"] = "child"
        sorted_by_size[-1]["role"] = "elder"

        for p in sorted_by_size[1:-1]:
            p["role"] = "adult"

        return people

    people = assign_roles(people)

    # -------------------------------
    # AGE CORRECTION (REALISTIC)
    # -------------------------------
    def correct_age(p, people):

        role = p["role"]
        max_area = max([x["face_area"] for x in people])
        ratio = p["face_area"] / max_area

        if role == "child":
            if ratio < 0.3:
                return np.random.randint(3, 6)
            elif ratio < 0.5:
                return np.random.randint(6, 10)
            else:
                return np.random.randint(10, 14)

        if role == "adult":
            return max(22, min(p["age"], 45))

        if role == "elder":
            return max(50, p["age"])

        return p["age"]

    for p in people:
        p["corrected_age"] = correct_age(p, people)

    # -------------------------------
    # DRAW IMAGE
    # -------------------------------
    labeled_img = img.copy()

    for p in people:
        x, y, w, h = p["box"]
        center = (x + w // 2, y + h // 2)
        radius = int(max(w, h) / 2)

        cv2.circle(labeled_img, center, radius, (0, 0, 255), 2)
        cv2.putText(labeled_img, f"ID {p['person_id']}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Image")
        st.image(original_img, channels="BGR")

    with col2:
        st.write("Detected Faces")
        st.image(labeled_img, channels="BGR")

    # -------------------------------
    # DISPLAY PEOPLE
    # -------------------------------
    st.subheader("Detected Individuals")

    for p in people:
        st.write(
            f"ID {p['person_id']} → Age: {p['corrected_age']} | Role: {p['role']} | Gender: {p['gender']} | Emotion: {p['emotion']}"
        )

    # -------------------------------
    # RELATIONSHIP FUNCTION
    # -------------------------------
    def predict_relationship(p1, p2):
        r1, r2 = p1["role"], p2["role"]

        if ("elder" in [r1, r2]) and ("child" in [r1, r2]):
            return "Grandparent–Child"

        if ("adult" in [r1, r2]) and ("child" in [r1, r2]):
            return "Parent–Child"

        if ("adult" in [r1, r2]) and ("elder" in [r1, r2]):
            return "Parent–Grandparent"

        if r1 == r2:
            if r1 == "child":
                return "Siblings"
            if r1 == "adult":
                return "Peers"
            if r1 == "elder":
                return "Elderly Relation"

        return "Uncertain"

    # -------------------------------
    # BUILD RELATIONSHIPS
    # -------------------------------
    relationships = []

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            rel = predict_relationship(people[i], people[j])
            relationships.append((people[i]["person_id"], people[j]["person_id"], rel))

    # -------------------------------
    # DISPLAY RELATIONSHIPS
    # -------------------------------
    st.subheader("Relationships")

    for r in relationships:
        st.write(f"P{r[0]} ↔ P{r[1]} → {r[2]}")

    # -------------------------------
    # GRAPH
    # -------------------------------
    st.subheader("Relationship Graph")

    G = nx.Graph()

    for p in people:
        G.add_node(p["person_id"])

    for r in relationships:
        G.add_edge(r[0], r[1], label=r[2])

    pos = nx.spring_layout(G)

    edge_labels = nx.get_edge_attributes(G, 'label')

    fig = plt.figure()
    nx.draw(G, pos, with_labels=True, node_size=2500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)