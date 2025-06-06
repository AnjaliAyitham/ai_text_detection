# Generated by Django 4.2.18 on 2025-02-12 13:19

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UserRegistrationModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, validators=[django.core.validators.RegexValidator(message='Name must contain only letters', regex='^[a-zA-Z]+$')])),
                ('loginid', models.CharField(max_length=100, validators=[django.core.validators.RegexValidator(message='Login ID must contain only letters', regex='^[a-zA-Z]+$')])),
                ('password', models.CharField(max_length=100, validators=[django.core.validators.RegexValidator(message='Password must contain at least one number, one uppercase, one lowercase letter, and at least 8 characters', regex='(?=.*\\d)(?=.*[a-z])(?=.*[A-Z]).{8,}')])),
                ('mobile', models.CharField(max_length=10, validators=[django.core.validators.RegexValidator(message='Mobile must start with 5, 6, 7, 8, or 9 and be 10 digits long', regex='^[6789][0-9]{9}$')])),
                ('email', models.EmailField(max_length=100, validators=[django.core.validators.EmailValidator(message='Enter a valid email address')])),
                ('locality', models.CharField(max_length=100)),
                ('address', models.TextField(max_length=250)),
                ('status', models.CharField(default='waiting', max_length=100)),
            ],
        ),
    ]
