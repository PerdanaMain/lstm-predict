from database import get_connection
from log import print_log


def get_parts():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT id, part_name, type_id FROM pf_parts")
        tags = cur.fetchall()
        cur.close()
        conn.close()
        return tags
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred: {e}")


def get_part(part_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, part_name, type_id
            FROM pf_parts WHERE id = %s
            """,
            (part_id,),
        )
        part = cur.fetchall()
        cur.close()
        conn.close()
        return part
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred: {e}")


def get_values(part_id, features_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT date_time, value
            FROM dl_features_data
            WHERE part_id = %s AND features_id = %s
            ORDER BY date_time ASC;
            """,
            (part_id, features_id),
        )

        # cur.execute(
        #     """
        #     SELECT date_time, value
        #     FROM dl_features_data
        #     WHERE part_id = %s
        #     AND date_time NOT BETWEEN '2024-11-01' AND '2024-11-06'
        #     ORDER BY date_time ASC;
        #     """,
        #     (part_id,),
        # )

        values = cur.fetchall()
        cur.close()
        conn.close()
        return values
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred {e}")
