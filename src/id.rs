use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use ulid::Ulid;

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub struct VectorID(Ulid);

impl VectorID {
    pub fn new() -> Self {
        Self(Ulid::new())
    }

    pub fn as_ulid(&self) -> Ulid {
        self.0
    }
}

impl From<Ulid> for VectorID {
    fn from(value: Ulid) -> Self {
        Self(value)
    }
}

impl TryFrom<String> for VectorID {
    type Error = ulid::DecodeError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse::<Ulid>().map(Self)
    }
}

impl TryFrom<&str> for VectorID {
    type Error = ulid::DecodeError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse::<Ulid>().map(Self)
    }
}

impl fmt::Display for VectorID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Serialize for VectorID {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for VectorID {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct VectorIDVisitor;

        impl<'de> Visitor<'de> for VectorIDVisitor {
            type Value = VectorID;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string vector id")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                value.parse::<Ulid>().map(VectorID).map_err(E::custom)
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                value.parse::<Ulid>().map(VectorID).map_err(E::custom)
            }
        }

        deserializer.deserialize_string(VectorIDVisitor)
    }
}
