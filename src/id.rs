use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct VectorID(String);

impl VectorID {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for VectorID {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for VectorID {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for VectorID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for VectorID {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
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
                Ok(VectorID::from(value))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(VectorID::from(value))
            }
        }

        deserializer.deserialize_string(VectorIDVisitor)
    }
}
