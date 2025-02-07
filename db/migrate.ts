import * as dotenv from "dotenv";
import { drizzle } from "drizzle-orm/postgres-js";
import { migrate } from "drizzle-orm/postgres-js/migrator";
import postgres from "postgres";

dotenv.config();

const runMigrations = async () => {
  if (!process.env.POSTGRES_URL) {
    throw new Error("POSTGRES_URL is not defined");
  }

  const connection = postgres(process.env.POSTGRES_URL, { max: 1 });
  const db = drizzle(connection);

  console.log("Running migrations...");

  await migrate(db, {
    migrationsFolder: "lib/drizzle",
  });

  console.log("Migrations completed!");

  await connection.end();
};

runMigrations().catch((err) => {
  console.error("Error running migrations:", err);
  process.exit(1);
});
