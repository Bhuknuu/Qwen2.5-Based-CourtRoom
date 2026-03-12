import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class Courtroom {

    public static void main(String[] args) {
        String statement;
        int rounds;

        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter statement to debate: ");
        statement = scanner.nextLine().trim();

        System.out.print("Number of max rounds: ");
        try { rounds = Integer.parseInt(scanner.nextLine().trim()); }
        catch (NumberFormatException e) {rounds=3; /*default*/ }

        if (statement.isEmpty()) {
            System.out.println("No Statement found, Exiting...");
            System.exit(1);
        }

        Orchestrator court = new Orchestrator(rounds);
        DebateSession result = court.debate(statement, true);

        System.out.println("\nDone ! | " + result.getCurrentRound() + " round(s) | " + result.getTranscript().size() + " total messages.");
    }
}

class Orchestrator {
    private final Critic critic;
    private final For forSide;
    private final Judge judge;
    private final int maxRounds;
    private final OllamaClient client;

    public Orchestrator(int maxRounds) {
        this.client = new OllamaClient("qwen2.5");
        this.critic = new Critic(client);
        this.forSide = new For(client);
        this.judge = new Judge(client);
        this.maxRounds = maxRounds;
    }

    public DebateSession debate(String statement, boolean verbose) {
        DebateSession session = new DebateSession(statement, maxRounds);
        String line = "_ + .".repeat(20);

        // ── opening: clarify what FOR and AGAINST mean for this specific statement
        String[] sides = clarifyStances(statement);
        String forMeans     = sides[0];
        String againstMeans = sides[1];

        if (verbose) {
            System.out.println("\n" + line);
            System.out.println("Statement : \"" + statement + "\"");
            System.out.println("FOR       : " + forMeans);
            System.out.println("AGAINST   : " + againstMeans);
            System.out.println("MaxRounds : " + maxRounds);
            System.out.println(line);
        }

        for (int round = 1; round <= maxRounds; round++) {
            if (verbose) System.out.println("\nRound:" + round);

            String cArg = critic.argue(session);
            session.addMessage(critic.getName(), cArg);
            if (verbose) System.out.println("\nAgainst:\n" + cArg);

            String fArg = forSide.argue(session);
            session.addMessage(forSide.getName(), fArg);
            if (verbose) System.out.println("\nFor:\n" + fArg);

            if (verbose) System.out.println("\nJudge...");

            // judge picks exactly one winner for this round — no scores, just a name
            String roundWinner = judge.pickRoundWinner(session);

            if (roundWinner.equals("AGAINST")) {
                session.addCriticPoint();
            } else {
                session.addForPoint();
            }

            if (verbose) {
                System.out.println("Round winner : " + roundWinner);
                System.out.println("Score        : Against " + session.getCriticPoints()
                                 + " — For " + session.getForPoints());
            }

            boolean finalRound  = (round == maxRounds);
            boolean pointsGap   = Math.abs(session.getCriticPoints() - session.getForPoints()) == maxRounds;

            // stop early only if one side has swept every round so far and it's decisive
            if (finalRound || pointsGap) {
                // final verdict: true/false on the statement + winner summary
                String verdict = judge.finalVerdict(session);
                String overallWinner = session.getCriticPoints() > session.getForPoints() ? "AGAINST" : "FOR";
                session.conclude(overallWinner, verdict);

                if (verbose) {
                    System.out.println("\n" + line);
                    System.out.println("Final Score : Against " + session.getCriticPoints()
                                     + " — For " + session.getForPoints());
                    System.out.println("Verdict     : " + verdict);
                    System.out.println("Winner      : " + overallWinner);
                    System.out.println(line);
                }
                break;
            } else {
                session.nextRound();
            }
        }return session;
    }

    // one small call at the start — asks the model what FOR and AGAINST mean
    // for this specific statement so the user knows what each side is arguing
    private String[] clarifyStances(String statement) {
        String system = "You clarify debate positions. Be brief and direct.";
        String prompt = "For the statement: \"" + statement + "\"\n"
                      + "Respond in EXACTLY this format, one line each, no extra text:\n"
                      + "FOR: <what the FOR side is arguing in one short phrase>\n"
                      + "AGAINST: <what the AGAINST side is arguing in one short phrase>";
        try {
            String response = client.chat(system, prompt);
            String forMeans     = "agrees with the statement";
            String againstMeans = "disagrees with the statement";
            for (String l : response.split("\n")) {
                if (l.startsWith("FOR:"))     forMeans     = l.substring(4).trim();
                if (l.startsWith("AGAINST:")) againstMeans = l.substring(8).trim();
            }
            return new String[]{forMeans, againstMeans};
        } catch (IOException e) {
            return new String[]{"agrees with the statement", "disagrees with the statement"};
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
//  MESSAGE
//  one thing someone said. immutable. the courtroom record can't be edited.
// ──────────────────────────────────────────────────────────────────────────────

class Message {
    private final String speaker;
    private final String content;
    private final int round;

    public Message(String speaker, String content, int round) {
        this.speaker = speaker;
        this.content = content;
        this.round = round;
    }

    public String getSpeaker(){return speaker;}
    public String getContent(){return content;}
    public int getRound(){return round;}

    @Override
    public String toString() {
        return "[R" + round + "] " + speaker + ": " + content;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
//  DEBATE SESSION
//  shared memory. every agent reads from here. no one owns it.
//  all state lives here so agents can stay stateless and swappable.
// ──────────────────────────────────────────────────────────────────────────────

class DebateSession {
    private final String statement;
    private final List<Message> transcript;
    private final int maxRounds;
    private int currentRound;
    private boolean concluded;
    private String winner;
    private String verdictText;

    // integer scoreboard — no string parsing, just increment
    private int criticPoints;
    private int forPoints;

    public DebateSession(String statement, int maxRounds) {
        this.statement = statement;
        this.maxRounds = maxRounds;
        this.transcript = new ArrayList<>();
        this.currentRound = 1;
        this.concluded = false;
        this.criticPoints = 0;
        this.forPoints = 0;
    }

    public void addMessage(String speaker, String content) {
        transcript.add(new Message(speaker, content, currentRound));
    }

    public void addCriticPoint(){criticPoints++;}
    public void addForPoint(){forPoints++;}

    public void nextRound(){currentRound++;}

    public void conclude(String winner, String verdictText) {
        this.winner = winner;
        this.verdictText = verdictText;
        this.concluded = true;
    }

    // dumps the whole transcript as plain text for the LLM's context window
    public String buildTranscriptContext() {
        if (transcript.isEmpty()) return "No arguments yet. Opening round.";
        StringBuilder sb = new StringBuilder();
        for (Message m : transcript) sb.append(m.toString()).append("\n\n");
        return sb.toString().trim();
    }

    // builds only the messages from one side — used for verdict summary
    public String buildSideTranscript(String speakerName) {
        StringBuilder sb = new StringBuilder();
        for (Message m : transcript)
            if (m.getSpeaker().equals(speakerName)) sb.append(m.getContent()).append("\n\n");
        return sb.toString().trim();
    }

    public String getStatement(){return statement;}
    public int getCurrentRound(){return currentRound;}
    public int getMaxRounds(){return maxRounds;}
    public boolean isConcluded(){return concluded;}
    public String getWinner(){return winner;}
    public String getVerdictText(){return verdictText;}
    public int getCriticPoints(){return criticPoints;}
    public int getForPoints(){return forPoints;}
    public List<Message> getTranscript(){return Collections.unmodifiableList(transcript);}
}

// ──────────────────────────────────────────────────────────────────────────────
//  AGENT  (interface)
//  the whole Strategy Pattern is just this. one method
// ──────────────────────────────────────────────────────────────────────────────

interface Agent {
    String getName();
    String getSystemPrompt();
    String argue(DebateSession session);
}

// ──────────────────────────────────────────────────────────────────────────────
//  OLLAMA CLIENT
//  sends a POST to localhost:11434, gets text back. that's it.
//  NOTE to myself: change the model string in Orchestrator if running something else.
// ──────────────────────────────────────────────────────────────────────────────

class OllamaClient {
    private static final String URL = "http://localhost:11434/api/chat";
    private static final int TIMEOUT = 120;

    private final String model;
    private final HttpClient http;

    public OllamaClient(String model) {
        this.model = model;
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    }

    public String chat(String system, String user) throws IOException {
        String body = "{"
            + "\"model\":\"" + model + "\","
            + "\"stream\":false,"
            + "\"messages\":["
            +   "{\"role\":\"system\",\"content\":\"" + esc(system) + "\"},"
            +   "{\"role\":\"user\",\"content\":\"" + esc(user) + "\"}"
            + "]}";

        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(URL))
                .timeout(Duration.ofSeconds(TIMEOUT))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> res;
        try {
            res = http.send(req, HttpResponse.BodyHandlers.ofString());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Request interrupted", e);
        }

        if (res.statusCode() != 200)
            throw new IOException("Ollama returned HTTP " + res.statusCode());

        return parseContent(res.body());
    }

    // hand-rolled JSON parser. yes I know. adding jackson felt like cheating.
    private String parseContent(String json) throws IOException {
        String key = "\"content\":\"";
        int i = json.indexOf(key);
        if (i == -1) throw new IOException("No content field in response");
        i += key.length();

        StringBuilder sb = new StringBuilder();
        for (; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c=='\\' && i+1 < json.length()) {
                char n = json.charAt(++i);
                switch (n) {
                    case '"' -> sb.append('"');
                    case 'n' -> sb.append('\n');
                    case 't' -> sb.append('\t');
                    case '\\'-> sb.append('\\');
                    default  -> sb.append(n);
                }
            } else if (c == '"') {
                break;
            } else {
                sb.append(c);
            }
        }
        return sb.toString().trim();
    }

    private String esc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
//  CRITIC
//  finds everything wrong with the idea.
// ──────────────────────────────────────────────────────────────────────────────

class Critic implements Agent {
    private static final String PERSONA =
        """
        You are the Critic in a structured adversarial debate.
        Argue AGAINST the statement. Expose its flaws, risks, and gaps.
        Never concede. Counter what For said directly.
        4-6 sentences max.""";

    private final OllamaClient client;
    public Critic(OllamaClient client) { this.client = client; }

    @Override
    public String getName(){return "Against"; }

    @Override
    public String getSystemPrompt(){return PERSONA; }

    @Override
    public String argue(DebateSession s) {
        String prompt = "Statement: \"" + s.getStatement() + "\"\n\n" + "Transcript so far:\n" + s.buildTranscriptContext() + "\n\n";
        try { return client.chat(PERSONA, prompt); }
        catch (IOException e) { return "Against offline: " + e.getMessage(); }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
//  FOR
//  finds the good in everything. the optimist.
// ──────────────────────────────────────────────────────────────────────────────

class For implements Agent {
    private static final String PERSONA =
        """
        You are For in a structured adversarial debate.
        Argue IN FAVOUR of the statement. Defend it, justify it, show its merit.
        Never concede. Directly counter what the Critic said.
        4-6 sentences max.""";

    private final OllamaClient client;
    public For(OllamaClient client) {this.client = client; }

    @Override
    public String getName(){ return "For"; }

    @Override
    public String getSystemPrompt() {return PERSONA;}

    @Override
    public String argue(DebateSession s) {
        String prompt = "Statement: \"" + s.getStatement() + "\"\n\n" + "Transcript so far:\n" + s.buildTranscriptContext() + "\n\n";
        try {return client.chat(PERSONA, prompt); }
        catch (IOException e) {return "For offline: " + e.getMessage(); }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
//  JUDGE
//  two jobs, two methods:
//    pickRoundWinner  — reads this round's two arguments, returns "AGAINST" or "FOR"
//    finalVerdict     — reads full transcript, returns true/false on the statement
//                       + a summary of why the winning side won
// ──────────────────────────────────────────────────────────────────────────────

class Judge {

    private final OllamaClient client;
    public Judge(OllamaClient client) { this.client = client; }

    // called once per round. returns exactly "AGAINST" or "FOR". nothing else needed.
    public String pickRoundWinner(DebateSession s) {
        String system =
            """
            You are the Judge in a structured debate. Read the latest round of arguments and decide who argued better.
            Respond with EXACTLY one word: AGAINST or FOR. No punctuation, no explanation.""";
        String prompt = "Statement: \"" + s.getStatement() + "\"\n\n"
                      + "Full transcript:\n" + s.buildTranscriptContext() + "\n\n"
                      + "Who won this round? Reply with one word only: AGAINST or FOR";
        try {
            String response = client.chat(system, prompt).trim().toUpperCase();
            // if the model rambles, scan for the first valid keyword
            if (response.contains("AGAINST")) return "AGAINST";
            if (response.contains("FOR"))    return "FOR";
            return "FOR"; // fallback — shouldn't happen often
        } catch (IOException e) {
            return "FOR"; // silent fallback, debate continues
        }
    }

    // called once at the very end. returns the verdict as a plain sentence.
    // format: "TRUE: <summary>" or "FALSE: <summary>"
    // TRUE  = the statement holds after debate
    // FALSE = the statement does not hold after debate
    public String finalVerdict(DebateSession s) {
        String winnerName  = s.getCriticPoints() > s.getForPoints() ? "Against" : "For";
        String winnerSide  = s.getCriticPoints() > s.getForPoints() ? "AGAINST" : "FOR";
        String winnerArgs  = s.buildSideTranscript(winnerName);

        String system =
            """
            You are the Judge delivering a final verdict on a debated statement.
            Respond in EXACTLY this format, one line only:
            TRUE: <one sentence summary of why the statement holds, using the winning side's arguments>
            or
            FALSE: <one sentence summary of why the statement does not hold, using the winning side's arguments>
            No extra text. The summary must reflect the actual arguments made.""";

        String prompt = "Statement: \"" + s.getStatement() + "\"\n\n"
                      + "The winning side was: " + winnerSide + "\n\n"
                      + "Winning side's arguments across all rounds:\n" + winnerArgs + "\n\n"
                      + "Final score: Against " + s.getCriticPoints() + " — For " + s.getForPoints() + "\n\n"
                      + "Deliver the final verdict.";
        try {
            return client.chat(system, prompt).trim();
        } catch (IOException e) {
            return "Verdict unavailable: " + e.getMessage();
        }
    }
}