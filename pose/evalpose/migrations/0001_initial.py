# Generated by Django 4.2.19 on 2025-03-03 10:59

from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="EvalSession",
            fields=[
                (
                    "session_id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "等待处理"),
                            ("processing", "处理中"),
                            ("completed", "已完成"),
                            ("failed", "失败"),
                            ("cancelled", "已取消"),
                        ],
                        default="pending",
                        max_length=20,
                    ),
                ),
                ("score", models.FloatField(blank=True, null=True)),
                ("error_message", models.TextField(blank=True, null=True)),
                ("dtw_distance", models.FloatField(blank=True, null=True)),
                ("similarity_score", models.FloatField(blank=True, null=True)),
                ("frame_scores", models.JSONField(blank=True, null=True)),
                (
                    "report_path",
                    models.CharField(blank=True, max_length=255, null=True),
                ),
                ("frame_data", models.JSONField(blank=True, null=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="VideoFile",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("file", models.FileField(upload_to="videos/")),
                (
                    "video_type",
                    models.CharField(
                        choices=[("standard", "标准视频"), ("exercise", "练习视频")],
                        max_length=20,
                    ),
                ),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
                (
                    "session",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="videos",
                        to="evalpose.evalsession",
                    ),
                ),
            ],
            options={
                "unique_together": {("session", "video_type")},
            },
        ),
    ]
