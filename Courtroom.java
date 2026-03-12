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
        String proposition;
        int rounds;

        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter proposition to debate: ");
        proposition = scanner.nextLine().trim();

        System.out.print("Number of max rounds: ");
        try { rounds = Integer.parseInt(scanner.nextLine().trim()); }
        catch (NumberFormatException e) {rounds=3; /*default*/ }

        if (proposition.isEmpty()) {
            System.out.println("No Proposition found, Exiting...");
            System.exit(1);
        }

        Orchestrator court = new Orchestrator(rounds);
        DebateSession result = court.debate(proposition, true);

        System.out.println("\nDone ! | " + result.getCurrentRound() + " round(s) | " + result.getTranscript().size() + " total messages.");
    }
}

class Orchestrator {
    private final Prosecutor prosecutor;
    private final DefenseAttorney defense;
    private final Judge judge;
    private final int maxRounds;

    public Orchestrator(int maxRounds) { //Constructor
        OllamaClient client = new OllamaClient("qwen2.5");
        this.prosecutor = new Prosecutor(client);
        this.defense = new DefenseAttorney(client);
        this.judge = new Judge(client);
        this.maxRounds = maxRounds;
    }

    public DebateSession debate(String proposition, boolean verbose) {
        DebateSession session = new DebateSession(proposition, maxRounds);
        String line="_+._".repeat(20);
        if (verbose) {
            System.out.println("\n" + line);
            System.out.println("Proposition : \"" + proposition + "\"");
            System.out.println("MaxRounds  : " + maxRounds);
            System.out.println(line);
        }

        for (int round = 1; round <= maxRounds; round++) {
            if (verbose) System.out.println("\nRound:" + round);

            String pArg = prosecutor.argue(session);
            session.addMessage(prosecutor.getName(), pArg);
            if (verbose) System.out.println("\nProsecutor:\n" + pArg);

            String dArg = defense.argue(session);
            session.addMessage(defense.getName(), dArg);
            if (verbose) System.out.println("\nDefense:\n" + dArg);

            if (verbose) System.out.println("\nJudge...");
            String judgeOut = judge.argue(session);

            if (verbose) {
                System.out.println(">>> " + Judge.assessment(judgeOut));
                System.out.println("Prosecution" + Judge.score(judgeOut, "ProsecutorScore") + " | Defense " + Judge.score(judgeOut, "DefenseScore"));
            }

            if (!Judge.needsRebuttal(judgeOut) || round == maxRounds) {
                session.conclude(Judge.winner(judgeOut), Judge.verdict(judgeOut));
                if (verbose) {
                    System.out.println("\n" + line);
                    System.out.println("Verdict : " + session.getVerdictText());
                    System.out.println("Winner  : " + session.getWinner());
                    System.out.println(line);
                }
                // break;
            } else {
                if (verbose) System.out.println("Rebuttal : " + Judge.verdict(judgeOut));
                session.nextRound();
            }
        }return session;
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
    private final String proposition;
    private final List<Message> transcript;
    private final int maxRounds;
    private int currentRound;
    private boolean concluded;
    private String winner;
    private String verdictText;

    public DebateSession(String proposition, int maxRounds) {
        this.proposition = proposition;
        this.maxRounds = maxRounds;
        this.transcript = new ArrayList<>();
        this.currentRound = 1;
        this.concluded = false;
    }

    public void addMessage(String speaker, String content) {
        transcript.add(new Message(speaker, content, currentRound));
    }

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

    public String getProposition(){return proposition;}
    public int getCurrentRound(){return currentRound;}
    public int getMaxRounds(){return maxRounds;}
    public boolean isConcluded(){return concluded;}
    public String getWinner(){return winner;}
    public String getVerdictText(){return verdictText;}
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
                .header("Content_Type", "application/json")
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
//  PROSECUTOR
//  finds everything wrong with the idea.
// ──────────────────────────────────────────────────────────────────────────────

class Prosecutor implements Agent {
    private static final String PERSONA =
        """
        You are the Prosecutor in a structured adversarial debate. 
        Argue AGAINST the proposition. Expose its flaws, risks, and gaps.
        Never concede. Counter what the Defense said directly.
        4-6 sentences max. End with "\nPROSECUTION POINT: <core objection>". """;

    private final OllamaClient client;
    public Prosecutor(OllamaClient client) { this.client = client; }

    @Override 
    public String getName(){return "Prosecutor"; }
    
    @Override 
    public String getSystemPrompt(){return PERSONA; }

    @Override
    public String argue(DebateSession s) {
        String prompt = "Proposition: \"" + s.getProposition() + "\"\n\n" + "Transcript so far:\n" + s.buildTranscriptContext() + "\n\n";
        try { return client.chat(PERSONA, prompt); }
        catch (IOException e) { return "Prosecutor offline: " + e.getMessage(); }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
//  DEFENSE ATTORNEY
//  finds the good in everything. the optimist.
// ──────────────────────────────────────────────────────────────────────────────

class DefenseAttorney implements Agent {

    private static final String PERSONA =
        """
        You are the Defense Attorney in a structured adversarial debate.
        Argue IN FAVOUR of the proposition. Defend it, justify it, show its merit.
        Never concede. Directly counter what the Prosecutor said.
        4-6 sentences max. End with "\nDEFENSE POINT: <strongest justification>".""";

    private final OllamaClient client;
    public DefenseAttorney(OllamaClient client) {this.client = client; }

    @Override 
    public String getName(){ return "Defense Attorney"; }
    
    @Override 
    public String getSystemPrompt() {return PERSONA;}

    @Override
    public String argue(DebateSession s) {
        String prompt = "Proposition: \"" + s.getProposition() + "\"\n\n" + "Transcript so far:\n" + s.buildTranscriptContext() + "\n\n";
        try {return client.chat(PERSONA, prompt); }
        catch (IOException e) {return "defense offline: " + e.getMessage(); }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
//  JUDGE
//  reads the whole fight, scores both sides, decides if we need another round.
//  strict output format so the orchestrator can parse it without doing regex.
// ──────────────────────────────────────────────────────────────────────────────

class Judge implements Agent {

    private static final String PERSONA =
        """
        You are the Judge in a structured debate. Be objective. Respond in EXACTLY this format, no extra text:
        ProsecutorScore: <1-10>
        DefenseScore: <1-10>
        NeedsRebuttal: <YES or NO>
        Assessment: <one sentence>
        Verdict: <final verdict if NO, or the unresolved question if YES>
        Winner: <PROSECUTION, DEFENSE, DRAW, or PENDING>""";

    private final OllamaClient client;
    public Judge(OllamaClient client) { this.client = client; }

    @Override 
    public String getName(){return "Judge";}
    
    @Override 
    public String getSystemPrompt() {return PERSONA;}

    @Override
    public String argue(DebateSession s) {
        boolean last = s.getCurrentRound() >= s.getMaxRounds();
        String prompt = "Prosecutor: \"" + s.getProposition() + "\"\n\n"
                      + "Transcript:\n" + s.buildTranscriptContext() + "\n\n"
                      + "Round " + s.getCurrentRound() + "/" + s.getMaxRounds() + ". "
                      + (last ? "Final round - verdict required. NeedsRebuttal must be NO." : "")
                      + "\nEvaluate.";
        try { return client.chat(PERSONA, prompt); }
        catch (IOException e) {
            return "Error in output File, Retry !";
        }
    }

    static boolean needsRebuttal(String r) {
        for (String l : r.split("\n"))
            if (l.startsWith("NeedsRebuttal:")) return l.contains("YES");
        return false;
    }
    static String verdict(String r) {
        for (String l : r.split("\n"))
            if (l.startsWith("Verdict:")) return l.substring(8).trim();
        return "No Verdict";
    }
    static String winner(String r) {
        for (String l : r.split("\n"))
            if (l.startsWith("Winner:")) 
                return l.substring(7).trim();
        return "DRAW";
    }
    static String assessment(String r) {
        for (String l : r.split("\n"))
            if (l.startsWith("Assessment:")) 
                return l.substring(11).trim();
        return "";
    }
    static int score(String r, String tag) {
        for (String l : r.split("\n"))
            if (l.startsWith(tag))
                try { return Integer.parseInt(l.substring(tag.length() + 1).trim()); }
                catch (NumberFormatException ignored) { return 0; }
        return 0;
    }
}
