'''Stores data model definitions. Even without a database.'''

from typing import Optional, Literal

import yaml
from pydantic import BaseModel, HttpUrl, parse_obj_as


class Contact(BaseModel):
    '''Represents contact record.'''

    key: str
    map: Literal['/', ':']
    value: str
    url: str


with open('data/contacts.yaml', encoding='utf-8') as file:
    _ = yaml.full_load(file)
    contacts = parse_obj_as(list[Contact], _['contacts'])
    links = parse_obj_as(list[Contact], _['links'])


class CVRecord(BaseModel):
    '''Represents CV record.'''

    organization: list[str]
    url: HttpUrl
    location: dict
    start_date: str
    end_date: str
    title: Optional[str]
    status: Optional[str]
    occupation: list[str]


with open('data/cv.yaml', encoding='utf-8') as file:
    _ = yaml.full_load(file)
    education = parse_obj_as(list[CVRecord], _['education'])
    experience = parse_obj_as(list[CVRecord], _['experience'])
