val sparkVersion = "2.4.0"

scalaVersion := "2.12.8"

resolvers ++= Seq(
  "apache-snapshots" at "http://repository.apache.org/snapshots/"
)

lazy val examples: Project = Project(
  "spark-ml-examples",
  file(".")
).settings(
    description := "Spark Processing",
    version := "0.0.1-SNAPSHOT",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion
    )
  )
