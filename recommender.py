#BUILD game text for similarity
def build_game_text(game_post):
    parts = []

    for hero in game_post.hero:
        parts.append(hero.game_title or "")
        parts.append(hero.game_desc_short or "")

    for story in game_post.storyline:
        parts.append(story.paragraphs or "")

    for gameplay in game_post.gameplay:
        parts.append(gameplay.paragraph or "")
        parts.append(gameplay.gameplay_title or "")

    for mechanic in game_post.mechanics:
        parts.append(mechanic.mechanic_text or "")

    for mode in game_post.modes:
        parts.append(mode.mode_title or "")
        parts.append(mode.mode_titledesc or "")

    for info in game_post.game_info:
        parts.append(info.genres or "")
        parts.append(info.platforms or "")

    return " ".join(parts)


#Loading all PUBLISHED game pages and preparing their text for recommendations
def load_game_corpus(session):
    games = session.query(GamePost).filter(
        GamePost.status == "published"
    ).all()

    texts = {}
    for g in games:
        texts[g.game_post_id] = build_game_text(g)

    return texts

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Converting text into numbers
def compute_similarity_matrix(game_texts):
    game_ids = list(game_texts.keys())
    corpus = list(game_texts.values())

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(corpus)

    sim_matrix = cosine_similarity(tfidf)

    return game_ids, sim_matrix

def get_user_state(interaction_count):
    if interaction_count < 5:
        return "cold"
    elif interaction_count < 30:
        return "warm"
    return "active"


def recommend_more_games(
    session,
    user_id,
    current_game_id,
    top_n=5
):
    # Load user interactions
    interactions = session.execute(
        sa.text("""
            SELECT game_post_id, interaction_type
            FROM interactions
            WHERE user_id = :uid
        """),
        {"uid": user_id}
    ).fetchall()

    interaction_count = len(interactions)
    user_state = get_user_state(interaction_count)

    # Load similarity data
    game_texts = load_game_corpus(session)
    game_ids, sim_matrix = compute_similarity_matrix(game_texts)

    game_index = {gid: idx for idx, gid in enumerate(game_ids)}
    current_idx = game_index[current_game_id]

    scores = {}

    for gid in game_ids:
        if gid == current_game_id:
            continue
        scores[gid] = 0.0

    #Item-to-item similarity (always used)
    for gid in scores:
        scores[gid] += (
            sim_matrix[current_idx][game_index[gid]] *
            (0.6 if user_state != "cold" else 0.3)
        )

    #User-item affinity
    if user_state != "cold":
        for gid, itype in interactions:
            if itype == "like":
                scores[gid] += 0.3

    #Similar users (same role)
    if user_state == "active":
        similar_users = session.execute(
            sa.text("""
                SELECT DISTINCT i2.user_id
                FROM interactions i1
                JOIN interactions i2
                  ON i1.game_post_id = i2.game_post_id
                JOIN users u1 ON u1.user_id = i1.user_id
                JOIN users u2 ON u2.user_id = i2.user_id
                WHERE i1.user_id = :uid
                  AND u1.role = u2.role
                  AND i2.user_id != :uid
            """),
            {"uid": user_id}
        ).fetchall()

        for (sid,) in similar_users:
            liked = session.execute(
                sa.text("""
                    SELECT game_post_id
                    FROM interactions
                    WHERE user_id = :sid
                      AND interaction_type = 'like'
                """),
                {"sid": sid}
            ).fetchall()

            for (gid,) in liked:
                if gid in scores:
                    scores[gid] += 0.25

    #Trending boost
    trending = session.execute(
        sa.text("""
            SELECT game_post_id, trending_score
            FROM trending
        """)
    ).fetchall()

    for gid, score in trending:
        if gid in scores:
            scores[gid] += (
                0.2 if user_state == "cold" else 0.05
            ) * score

    #Editorial picks
    if user_state == "cold":
        picks = session.execute(
            sa.text("""
                SELECT game_post_id
                FROM editorial_picks
            """)
        ).fetchall()

        for (gid,) in picks:
            if gid in scores:
                scores[gid] += 0.4

    #Final ranking
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]