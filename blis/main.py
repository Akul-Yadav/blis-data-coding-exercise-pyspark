"""PySpark transformations for the BLIS coding exercise."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


APP_NAME = "coding_example_run"
MICROSECONDS_IN_SECOND = 1_000_000
SESSION_GAP_SECONDS = 3 * 60 * 60


def _with_event_time(input_df: DataFrame) -> DataFrame:
    """Add a Spark timestamp made from the epoch-microseconds column."""
    return input_df.withColumn(
        "event_time",
        (F.col("timestamp").cast("double") / F.lit(MICROSECONDS_IN_SECOND)).cast(
            "timestamp"
        ),
    )


def filter_invalid_records(input_df: DataFrame) -> DataFrame:
    """Remove invalid activities and every duplicated user/timestamp group."""
    same_user_and_time = Window.partitionBy("user_id", "timestamp")

    return (
        input_df
        .filter(F.col("web_activity").isNotNull())
        .filter(F.col("web_activity") != "")
        .filter(F.col("web_activity").rlike(r"^[A-Za-z0-9|,%._=\-]+$"))
        .withColumn("records_at_same_time", F.count("*").over(same_user_and_time))
        .filter(F.col("records_at_same_time") == 1)
        .drop("records_at_same_time")
    )


def add_activity_lookup(input_df: DataFrame, lookup_df: DataFrame) -> DataFrame:
    """Add the exact pipe-delimited activity match, or code 999 if none exists."""
    filtered_input = filter_invalid_records(input_df).alias("events")
    lookup = lookup_df.alias("lookup")

    # Token matching means 'search' matches, while 'search1' does not.
    exact_activity_match = F.expr(
        "array_contains(split(events.web_activity, '[|]'), lookup.activity)"
    )

    return (
        filtered_input.join(lookup, exact_activity_match, "left")
        .select(
            "events.*",
            F.coalesce(F.col("lookup.activity_code").cast("int"), F.lit(999)).alias(
                "activity_code"
            ),
        )
    )


def get_top_3_active_hours(input_df: DataFrame) -> DataFrame:
    """Return the three hours having the most valid events on each calendar day."""
    filtered_input = _with_event_time(filter_invalid_records(input_df))

    hourly_counts = (
        filtered_input
        .withColumn("day", F.to_date("event_time"))
        .withColumn("hour", F.hour("event_time"))
        .groupBy("day", "hour")
        .agg(F.count("*").alias("activity_count"))
    )

    rank_for_each_day = Window.partitionBy("day").orderBy(
        F.desc("activity_count"), F.asc("hour")
    )

    return (
        hourly_counts
        .withColumn("rank", F.row_number().over(rank_for_each_day))
        .filter(F.col("rank") <= 3)
        .drop("rank")
        .select("hour", "day", "activity_count")
    )


def get_top_2_activity_codes(input_df: DataFrame, lookup_df: DataFrame) -> DataFrame:
    """Return the two most frequent activity codes on each calendar day."""
    input_with_activity_code = _with_event_time(
        add_activity_lookup(input_df, lookup_df)
    )

    activity_counts = (
        input_with_activity_code
        .withColumn("day", F.to_date("event_time"))
        .groupBy("day", "activity_code")
        .agg(F.count("*").alias("activity_count"))
    )

    rank_for_each_day = Window.partitionBy("day").orderBy(
        F.desc("activity_count"), F.asc("activity_code")
    )

    return (
        activity_counts
        .withColumn("rank", F.row_number().over(rank_for_each_day))
        .filter(F.col("rank") <= 2)
        .drop("rank")
        .select("activity_code", "day", "activity_count")
    )


def get_activity_sessions_per_user(
    input_df: DataFrame, lookup_df: DataFrame
) -> DataFrame:
    """Create user sessions, starting a new one after a gap of at least 3 hours."""
    input_with_activity_code = _with_event_time(
        add_activity_lookup(input_df, lookup_df)
    )

    input_with_activity_code = input_with_activity_code.withColumn(
        "event_time_micros", F.col("timestamp").cast("long")
    )

    user_timeline = Window.partitionBy("user_id").orderBy("event_time_micros")
    user_timeline_to_current = user_timeline.rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )

    sessionized = (
        input_with_activity_code
        .withColumn(
            "previous_event_time_micros",
            F.lag("event_time_micros").over(user_timeline),
        )
        .withColumn(
            "starts_new_session",
            F.when(F.col("previous_event_time_micros").isNull(), 1)
            .when(
                F.col("event_time_micros")
                - F.col("previous_event_time_micros")
                >= SESSION_GAP_SECONDS * MICROSECONDS_IN_SECOND,
                1,
            )
            .otherwise(0),
        )
        .withColumn(
            "session_id", F.sum("starts_new_session").over(user_timeline_to_current)
        )
    )

    return (
        sessionized
        .groupBy("user_id", "session_id")
        .agg(
            F.min("event_time").alias("session_start"),
            F.min("event_time_micros").alias("session_start_micros"),
            F.max("event_time_micros").alias("session_end_micros"),
        )
        .select(
            F.to_date("session_start").alias("session_start_date"),
            "user_id",
            (
                (F.col("session_end_micros") - F.col("session_start_micros"))
                / F.lit(60.0 * MICROSECONDS_IN_SECOND)
            ).alias("session_time_in_mins"),
        )
    )


def main():
    """Application entry point; tests call the transformation functions directly."""
    print("Entering app..")


if __name__ == "__main__":
    main()
