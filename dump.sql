--
-- PostgreSQL database dump
--

-- Dumped from database version 16.8
-- Dumped by pg_dump version 17.4

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

ALTER TABLE IF EXISTS ONLY "public"."acts" DROP CONSTRAINT IF EXISTS "acts_pkey";
ALTER TABLE IF EXISTS "public"."acts" ALTER COLUMN "id" DROP DEFAULT;
DROP SEQUENCE IF EXISTS "public"."acts_id_seq";
DROP TABLE IF EXISTS "public"."acts";
SET default_tablespace = '';

SET default_table_access_method = "heap";

--
-- Name: acts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."acts" (
    "id" integer NOT NULL,
    "name" character varying(255) NOT NULL,
    "text" "text" NOT NULL,
    "year" integer NOT NULL,
    "act_number" character varying(50),
    "ministry" character varying(255),
    "embedding" double precision[],
    "amended" boolean
);


--
-- Name: acts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE "public"."acts_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: acts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE "public"."acts_id_seq" OWNED BY "public"."acts"."id";


--
-- Name: acts id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."acts" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."acts_id_seq"'::"regclass");


--
-- Data for Name: acts; Type: TABLE DATA; Schema: public; Owner: -
--

COPY "public"."acts" ("id", "name", "text", "year", "act_number", "ministry", "embedding", "amended") FROM stdin;
\.


--
-- Name: acts_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('"public"."acts_id_seq"', 1, false);


--
-- Name: acts acts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."acts"
    ADD CONSTRAINT "acts_pkey" PRIMARY KEY ("id");


--
-- PostgreSQL database dump complete
--

